import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, Swish, and Scaling
# This version uses a Tiled GEMM approach for better performance on large matrices
# and fixes the functional correctness issue by handling weight layout correctly.
fused_gemm_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Tile size for the GEMM kernel
#define TILE_SIZE 32

// Kernel for fused GEMM + Swish + Scaling
__global__ void fused_gemm_swish_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int K, int N, float scaling_factor) {
    
    // Thread block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over all tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load a tile of input into shared memory
        int inputRow = row;
        int inputCol = t * TILE_SIZE + tx;
        if (inputRow < M && inputCol < K) {
            As[ty][tx] = input[inputRow * K + inputCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load a tile of weight into shared memory
        // Weight is (N, K). We need weight[col, t*TILE_SIZE + ty]
        int weightRow = col;
        int weightCol = t * TILE_SIZE + ty;
        if (weightRow < N && weightCol < K) {
            Bs[ty][tx] = weight[weightRow * K + weightCol];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write the result
    if (row < M && col < N) {
        // Add bias
        if (bias != nullptr) {
            sum += bias[col];
        }

        // Apply Swish activation: x * sigmoid(x)
        float sig = 1.0f / (1.0f + expf(-sum));
        float swish = sum * sig;

        // Apply scaling
        output[row * N + col] = swish * scaling_factor;
    }
}

torch::Tensor fused_gemm_swish_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor) {
    
    // Get dimensions
    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = weight.size(0); // Weight shape is (out_features, in_features)

    // Allocate output tensor
    auto output = torch::empty({M, N}, input.options());

    // Configure grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    fused_gemm_swish_scale_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() && bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        M, K, N, scaling_factor
    );

    // Check for launch errors
    C10_CUDA_CHECK(cudaGetLastError());

    return output;
}
"""

fused_gemm_swish_scale_cpp_source = (
    "torch::Tensor fused_gemm_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_gemm_swish_scale_op = load_inline(
    name="fused_gemm_swish_scale",
    cpp_sources=fused_gemm_swish_scale_cpp_source,
    cuda_sources=fused_gemm_swish_scale_source,
    functions=["fused_gemm_swish_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused GEMM + Swish + Scaling.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # We still define weights as Parameters so they are trained/loaded correctly
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scaling_factor = scaling_factor
        
        # Initialize weights (mimicking nn.Linear default initialization)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # Load the custom op
        self.fused_op = fused_gemm_swish_scale_op

    def forward(self, x):
        # Call the custom fused kernel
        return self.fused_op.fused_gemm_swish_scale_cuda(x, self.weight, self.bias, self.scaling_factor)