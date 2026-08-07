import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, scaling, hardtanh, and GELU
fused_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Constants for GELU approximation (using exact erf based GELU as per PyTorch default)
// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

__global__ void gemm_fused_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int M, int K, int N, 
    float scaling_factor, float hardtanh_min, float hardtanh_max) {
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices
    int row = by * blockDim.y + ty; // M dimension (batch size)
    int col = bx * blockDim.x + tx; // N dimension (output features)
    
    // Shared memory for input and weight tiles
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    float val = 0.0f;
    
    // Loop over K dimension in tiles
    for (int m = 0; m < (K + 31) / 32; ++m) {
        // Load input tile (M x K)
        // Input is row-major. We load a 32x32 tile.
        // As[ty][tx] corresponds to input[row][m*32 + tx]
        int k_idx_input = m * 32 + tx;
        if (row < M && k_idx_input < K) {
            As[ty][tx] = input[row * K + k_idx_input];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load weight tile (N x K)
        // Weight is (N, K) row-major. We need to multiply Input * W^T effectively in standard GEMM terms,
        // but here we compute Output = Input * Weight^T.
        // Let's look at the standard linear layer: y = xW^T + b.
        // W is (N, K). W^T is (K, N).
        // We want to compute dot product of row of X and row of W.
        // Wait, standard linear: y_i = sum_j x_j * W_{i, j}.
        // So we need column 'col' of W (which is row 'col' of the weight matrix passed).
        // We need to load a tile of W.
        // Bs[ty][tx] should correspond to W[col][m*32 + ty]?
        // Let's trace the indices for the multiplication.
        // We are computing C[row][col].
        // C[row][col] = sum_k A[row][k] * B[col][k] (where B is the weight matrix W).
        // So we need A's row and B's row.
        // A is loaded into As. As[ty][k] is the k-th element of the row tile.
        // B needs to be loaded. We need B[col][k].
        // In the loop, k iterates.
        // We load a tile of B. The tile covers rows [bx*32, bx*32+32] and cols [m*32, m*32+32].
        // Thread (tx, ty) loads B[bx*32 + tx][m*32 + ty].
        // This is coalesced access because adjacent tx access adjacent rows of B (adjacent memory locations if N is large).
        
        int k_idx_weight = m * 32 + ty;
        if (col < N && k_idx_weight < K) {
            // Weight is stored as (N, K)
            Bs[tx][ty] = weight[col * K + k_idx_weight];
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < 32; ++k) {
            // As[ty][k] holds input data
            // Bs[tx][k] holds weight data (transposed in shared memory relative to load)
            val += As[ty][k] * Bs[tx][k];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        // Add bias
        val += bias[col];
        
        // Scaling
        val *= scaling_factor;
        
        // Hardtanh
        val = fmaxf(hardtanh_min, fminf(val, hardtanh_max));
        
        // GELU (exact)
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        val = 0.5f * val * (1.0f + erff(val * 0.70710678118654752440f));
        
        output[row * N + col] = val;
    }
}

torch::Tensor gemm_fused_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float scaling_factor, 
    float hardtanh_min, 
    float hardtanh_max) {
    
    int M = input.size(0); // Batch size
    int K = input.size(1); // In features
    int N = weight.size(0); // Out features
    
    auto output = torch::zeros({M, N}, input.options());
    
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    
    gemm_fused_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        M, K, N, 
        scaling_factor, hardtanh_min, hardtanh_max);
        
    return output;
}
"""

fused_gemm_cpp_source = (
    "torch::Tensor gemm_fused_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor, float hardtanh_min, float hardtanh_max);"
)

# Compile the inline CUDA code
fused_gemm_op = load_inline(
    name="fused_gemm",
    cpp_sources=fused_gemm_cpp_source,
    cuda_sources=fused_gemm_source,
    functions=["gemm_fused_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a fused GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # We keep the standard Linear layer to hold the weights and bias.
        # The forward pass will use our custom CUDA kernel.
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_gemm = fused_gemm_op

    def forward(self, x):
        return self.fused_gemm.gemm_fused_cuda(
            x, 
            self.gemm.weight, 
            self.gemm.bias, 
            self.scaling_factor, 
            self.hardtanh_min, 
            self.hardtanh_max
        )