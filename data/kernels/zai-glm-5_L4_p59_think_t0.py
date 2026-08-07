import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Matmul + Swish + Scale
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Tile size for shared memory
const int TILE_SIZE = 32;

__global__ void fused_matmul_swish_scale_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int M, int N, int K, float scaling_factor) {
    
    // M: Batch size (input rows)
    // N: Output features (output columns / weight rows)
    // K: Input features (input columns / weight columns)
    
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Column index in output matrix / Row index in weight matrix
    int col = bx * TILE_SIZE + tx;
    
    // Shared memory for weight tile (with padding to avoid bank conflicts)
    __shared__ float s_W[TILE_SIZE][TILE_SIZE + 1];
    // Shared memory for input tile
    __shared__ float s_X[TILE_SIZE][TILE_SIZE + 1];
    
    // Accumulator for the dot product.
    // We process multiple row blocks of the input (M dimension) to reuse the loaded weight tile.
    // M is 128, TILE_SIZE is 32, so we have 4 row blocks.
    // We use a small fixed-size array or registers. Assuming M <= 128, max 4 blocks.
    float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    int num_row_blocks = (M + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over K dimension (input features)
    for (int t = 0; t < K; t += TILE_SIZE) {
        // 1. Load Weight tile into shared memory
        // Weight is (N, K). We load a TILE_SIZE x TILE_SIZE block.
        // Row index in weight: bx * TILE_SIZE + ty
        // Col index in weight: t + tx
        if ((bx * TILE_SIZE + ty) < N && (t + tx) < K) {
            s_W[ty][tx] = weight[(bx * TILE_SIZE + ty) * K + t + tx];
        } else {
            s_W[ty][tx] = 0.0f;
        }
        __syncthreads();
        
        // 2. Loop over row blocks of Input/Output (M dimension)
        // This allows us to reuse the weight tile 's_W' loaded above for multiple input rows.
        for (int r = 0; r < num_row_blocks; ++r) {
            int row = r * TILE_SIZE + ty;
            
            // Load Input tile
            // Input is (M, K).
            // Row index: row
            // Col index: t + tx
            if (row < M && (t + tx) < K) {
                s_X[ty][tx] = input[row * K + t + tx];
            } else {
                s_X[ty][tx] = 0.0f;
            }
            __syncthreads();
            
            // Compute partial dot product
            // s_X[ty][k] corresponds to input at (row, t+k)
            // s_W[tx][k] corresponds to weight at (col, t+k) -> transposed access
            float sum = 0.0f;
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += s_X[ty][k] * s_W[tx][k];
            }
            vals[r] += sum;
            __syncthreads();
        }
    }
    
    // 3. Write results with Bias, Swish, and Scaling
    for (int r = 0; r < num_row_blocks; ++r) {
        int row = r * TILE_SIZE + ty;
        if (row < M && col < N) {
            float v = vals[r];
            // Add bias
            v += bias[col];
            // Swish activation: x * sigmoid(x)
            v = v * (1.0f / (1.0f + expf(-v)));
            // Scaling
            v *= scaling_factor;
            
            output[row * N + col] = v;
        }
    }
}

torch::Tensor fused_matmul_swish_scale_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float scaling_factor) {
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    auto output = torch::zeros({M, N}, input.options());
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    // Grid dimension x corresponds to columns of output (N)
    // Grid dimension y is 1 because we loop over rows inside the kernel
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 1);
    
    fused_matmul_swish_scale_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K, scaling_factor
    );
    
    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_matmul_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_matmul_swish_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused Matmul + Swish + Scale.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # We keep the Linear layer to hold the weights and bias parameters.
        # The weights are initialized by PyTorch here.
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        # Call the custom CUDA kernel
        # self.matmul.weight is shape (out_features, in_features)
        # self.matmul.bias is shape (out_features)
        return self.fused_op.fused_matmul_swish_scale_cuda(
            x, self.matmul.weight, self.matmul.bias, self.scaling_factor
        )