import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Bias + ReLU
gemm_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiled matrix multiplication kernel with Bias and ReLU fusion
// Computes: Out = ReLU(Input * Weight^T + Bias)
// Input: (M, K), Weight: (N, K), Bias: (N,)
// Output: (M, N)

__global__ void gemm_bias_relu_kernel(const float* __restrict__ input, 
                                      const float* __restrict__ weight, 
                                      const float* __restrict__ bias, 
                                      float* __restrict__ output, 
                                      int M, int N, int K) {
    const int TILE_SIZE = 32;
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Shared memory for input tile (from Input matrix)
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    
    // Shared memory for weight tile (from Weight matrix)
    // Padding (+1) to avoid bank conflicts
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1]; 
    
    float sum = 0.0f;
    
    // Loop over K dimension by TILE_SIZE
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load a tile of Input into shared memory
        // Input is (M, K). We need element (row, t*TILE_SIZE + tx)
        int input_col = t * TILE_SIZE + tx;
        if (row < M && input_col < K) {
            s_A[ty][tx] = input[row * K + input_col];
        } else {
            s_A[ty][tx] = 0.0f;
        }
        
        // Load a tile of Weight into shared memory
        // Weight is (N, K). We need element (col, t*TILE_SIZE + ty)
        // We are computing Input * Weight^T, so we need row 'col' of Weight
        int weight_col = t * TILE_SIZE + ty;
        if (col < N && weight_col < K) {
            // Store transposed or directly?
            // We need to compute sum += Input[row, k] * Weight[col, k]
            // In the kernel loop: sum += s_A[ty][k] * s_B[tx][k] (if s_B stores row of Weight)
            // We load Weight[col, t*TILE_SIZE + ty] into s_B[tx][ty]
            // This puts the row of Weight into a row of s_B (indexed by tx)
            s_B[tx][ty] = weight[col * K + weight_col];
        } else {
            s_B[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[tx][k];
        }
        
        __syncthreads();
    }
    
    // Add bias and apply ReLU
    if (row < M && col < N) {
        float b = bias[col];
        sum += b;
        // ReLU: max(0, x)
        output[row * N + col] = sum > 0.0f ? sum : 0.0f;
    }
}

torch::Tensor gemm_bias_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // input: (M, K)
    // weight: (N, K)
    // bias: (N,)
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    auto output = torch::zeros({M, N}, input.options());
    
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_bias_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        M, N, K
    );
    
    return output;
}
"""

gemm_bias_relu_cpp_source = (
    "torch::Tensor gemm_bias_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
gemm_bias_relu = load_inline(
    name="gemm_bias_relu",
    cpp_sources=gemm_bias_relu_cpp_source,
    cuda_sources=gemm_bias_relu_source,
    functions=["gemm_bias_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for fused GEMM + Bias + ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        # We keep the weight parameter structure similar to nn.Linear for compatibility
        # but we will use our custom kernel.
        # Note: nn.Linear stores weights as (out_features, in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Initialize the custom op
        self.gemm_bias_relu = gemm_bias_relu

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        # Call the custom fused kernel
        return self.gemm_bias_relu.gemm_bias_relu_cuda(x, self.weight, self.bias)