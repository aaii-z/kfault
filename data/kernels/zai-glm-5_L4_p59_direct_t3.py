import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, Swish, and Scaling
# This implementation uses a tiled matrix multiplication approach (based on 32x32 tiles)
# to maximize data reuse and parallelism, which is essential for large matrices.
fused_gemm_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Tile size for the GEMM kernel
#define TILE_SIZE 32

// Sigmoid function
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fused GEMM (Tiled) + Swish + Scale Kernel
__global__ void fused_gemm_swish_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int K, int N, float scaling_factor) {
    
    // Shared memory for the tile of input and weight
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];

    // Calculate row and column indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over all tiles required to compute the element
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load a tile of input into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K) {
            ds_A[ty][tx] = input[row * K + t * TILE_SIZE + tx];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        // Load a tile of weight into shared memory
        // Weight is stored as (out_features, in_features) -> (N, K)
        // We are computing input * weight.T effectively, or row-major logic
        // We need weight[col, t*TILE_SIZE + ty]
        if (col < N && (t * TILE_SIZE + ty) < K) {
            ds_B[ty][tx] = weight[col * K + t * TILE_SIZE + ty];
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        // Add bias
        if (bias != nullptr) {
            sum += bias[col];
        }

        // Apply Swish activation: x * sigmoid(x)
        float swish = sum * sigmoid(sum);

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

    // Calculate grid and block dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    fused_gemm_swish_scale_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
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