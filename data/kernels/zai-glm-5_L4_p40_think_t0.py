import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear layer, scaling, and residual addition
fused_linear_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define tile size for shared memory tiling
#define TILE_SIZE 32

__global__ void fused_linear_scale_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int M, int N, int K, 
    float effective_scale) {
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices for the output matrix C (M x N)
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Shared memory for tiling input A (M x K) and weight B (N x K)
    // Padding (+1) to avoid shared memory bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    float val = 0.0f;
    
    // Loop over K dimension in steps of TILE_SIZE
    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load input tile (A) into shared memory
        // A is (M, K). We load A(row, k + tx)
        if (row < M && (k + tx) < K) {
            As[ty][tx] = input[row * K + k + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load weight tile (B) into shared memory
        // B is (N, K). We need to access B(col, k) which is row 'col' of B.
        // We load B(bx * TILE_SIZE + ty, k + tx) -> Bs[ty][tx]
        // This loads a tile of rows from B corresponding to the output columns.
        int weight_row = bx * TILE_SIZE + ty;
        if (weight_row < N && (k + tx) < K) {
            Bs[ty][tx] = weight[weight_row * K + k + tx];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        // We compute sum(A_row * B_col)
        // A_row is in As[ty][l]
        // B_col corresponds to row 'col' in B. 
        // We loaded rows of B into Bs. Row 'col' of B is in Bs[tx] (since tx corresponds to col offset).
        // So we access Bs[tx][l].
        for (int l = 0; l < TILE_SIZE; ++l) {
            val += As[ty][l] * Bs[tx][l];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        // Add bias
        val += bias[col];
        // Apply fused scaling: y = x * (1 + scaling_factor)
        val *= effective_scale;
        output[row * N + col] = val;
    }
}

torch::Tensor fused_linear_scale_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float effective_scale) {
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    auto output = torch::zeros({M, N}, input.options());
    
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_linear_scale_kernel<<<dimGrid, dimBlock>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K,
        effective_scale
    );
    
    return output;
}
"""

fused_linear_scale_cpp_source = (
    "torch::Tensor fused_linear_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float effective_scale);"
)

# Compile the inline CUDA code
fused_linear_scale = load_inline(
    name="fused_linear_scale",
    cpp_sources=fused_linear_scale_cpp_source,
    cuda_sources=fused_linear_scale_source,
    functions=["fused_linear_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # We keep the matmul layer to hold the weights and bias parameters
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_linear_scale

    def forward(self, x):
        # The operation is: y = (xW + b) * (1 + scaling_factor)
        effective_scale = 1.0 + self.scaling_factor
        
        # Call the custom CUDA kernel
        return self.fused_op.fused_linear_scale_cuda(
            x, 
            self.matmul.weight, 
            self.matmul.bias, 
            effective_scale
        )