import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, Min, and Subtraction
fused_gemm_min_sub_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Block size for tiling
#define BLOCK_SIZE 16

__global__ void fused_gemm_min_sub_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ bias, 
    float* __restrict__ C, 
    float constant, 
    int M, int N, int K) {
    
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column indices for C
    int row = bx * BLOCK_SIZE + ty;
    int col = by * BLOCK_SIZE + tx;

    // Accumulator for the dot product
    float c_val = 0.0f;

    // Shared memory for A and B tiles
    // A_tile is BLOCK_SIZE x BLOCK_SIZE
    // B_tile is BLOCK_SIZE x (BLOCK_SIZE + 1) to avoid bank conflicts during transpose access
    __shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Loop over all tiles in K dimension
    for (int t = 0; t < K; t += BLOCK_SIZE) {
        // Load A tile from global memory to shared memory
        // A is (M, K), we need A[row, t + tx]
        if (row < M && (t + tx) < K) {
            A_tile[ty][tx] = A[row * K + t + tx];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        // Load B tile from global memory to shared memory
        // B (weights) is (N, K) stored as (out_features, in_features)
        // We need B[col, t + tx] for the dot product A[row, :] . B[col, :]
        // To coalesce global memory reads, we want adjacent threads (tx) to read adjacent memory.
        // B[col, t + tx] is at index col * K + t + tx.
        // We load this into B_tile[ty][tx].
        // Note: ty corresponds to the row index of the block in C (which maps to col index of B).
        // Wait, 'col' is determined by 'by' and 'tx'.
        // Let's re-verify indexing.
        // Block (bx, by) computes a tile of C.
        // row = bx * BLOCK_SIZE + ty.
        // col = by * BLOCK_SIZE + tx.
        // We need B[col, k].
        // In the loading phase, thread (tx, ty) loads B_tile[ty][tx].
        // We want to load B[by * BLOCK_SIZE + ty, t + tx].
        // This corresponds to B_tile[ty][tx].
        // This is coalesced because adjacent tx read adjacent K elements.
        // And 'ty' varies across the block, selecting different rows of B (output features).
        
        if ((by * BLOCK_SIZE + ty) < N && (t + tx) < K) {
            B_tile[ty][tx] = B[(by * BLOCK_SIZE + ty) * K + t + tx];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Compute the partial dot product
        // We compute C[row, col] = sum_k A[row, k] * B[col, k]
        // A_tile[ty][k] holds A[row, t + k]
        // We need B[col, t + k].
        // B_tile holds B[by*16 + ty, t + tx].
        // We need B[col, ...]. col = by*16 + tx.
        // So we need B_tile[tx][k].
        // This accesses B_tile with the first index being tx.
        // This is a transpose access of the tile.
        // Due to padding (BLOCK_SIZE + 1), this is bank-conflict free.
        if (row < M && col < N) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                c_val += A_tile[ty][k] * B_tile[tx][k];
            }
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        // Add bias
        c_val += bias[col];
        
        // Fused Min and Subtraction
        // Logic: min(c_val, constant) - constant
        // Equivalent to: if (c_val < constant) c_val - constant; else 0;
        // Using ternary operator handles NaNs correctly (propagates NaN if c_val is NaN)
        c_val = (constant < c_val) ? constant : c_val;
        c_val -= constant;
        
        C[row * N + col] = c_val;
    }
}

// C++ wrapper function
torch::Tensor fused_gemm_min_sub_cuda(
    torch::Tensor x, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float constant) {
    
    // Dimensions
    int M = x.size(0);      // Batch size
    int K = x.size(1);      // In features
    int N = weight.size(0); // Out features

    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto out = torch::zeros({M, N}, options);

    // Grid and Block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    fused_gemm_min_sub_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        constant,
        M, N, K
    );

    return out;
}
"""

fused_gemm_min_sub_cpp_source = (
    "torch::Tensor fused_gemm_min_sub_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float constant);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_min_sub",
    cpp_sources=fused_gemm_min_sub_cpp_source,
    cuda_sources=fused_gemm_min_sub_source,
    functions=["fused_gemm_min_sub_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused GEMM, Min, and Subtraction.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        # We keep the linear layer to hold the weights and bias parameters.
        # We will use the weights directly in our custom kernel.
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.fused_op = fused_op

    def forward(self, x):
        # Extract the constant value. 
        # Note: This assumes constant is a scalar. If it requires grad, 
        # we detach it for the kernel launch. The kernel does not support 
        # autograd for the constant in this implementation.
        constant_val = self.constant.item()
        
        # Call the custom CUDA kernel
        return self.fused_op.fused_gemm_min_sub_cuda(x, self.linear.weight, self.linear.bias, constant_val)