import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, Bias, and Scaling
# This kernel computes: Output = (Input * Weight^T + Bias) * Scale
# Optimized with coalesced memory access and shared memory padding to avoid bank conflicts.
fused_gemm_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile size for the GEMM kernel
#define TILE_SIZE 32

__global__ void gemm_scale_bias_kernel(const float* __restrict__ a, const float* __restrict__ b, 
                                       const float* __restrict__ bias, const float* __restrict__ scale, 
                                       float* __restrict__ c, int N, int M, int K) {
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and Column indices for the output matrix C
    // Row index in C (and A)
    int row = bx * TILE_SIZE + ty;
    // Col index in C (and B, Scale, Bias)
    int col = by * TILE_SIZE + tx;

    // Accumulator for the dot product
    float c_val = 0.0f;

    // Shared memory for tiles of A and B
    // Pad Bs to avoid bank conflicts: TILE_SIZE + 1
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Loop over all tiles required to compute the element C[row][col]
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load a tile of A into shared memory
        // A is (N, K). We load A[row][t*32 + tx]
        // Coalesced: tx varies, accessing contiguous memory
        int a_col = t * TILE_SIZE + tx;
        if (row < N && a_col < K) {
            As[ty][tx] = a[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load a tile of B into shared memory
        // B is (M, K). We need B[col][k] for the computation.
        // We load B[by*32 + ty][t*32 + tx].
        // Coalesced: tx varies, accessing contiguous memory in B's row.
        int b_row = by * TILE_SIZE + ty;
        int b_col = t * TILE_SIZE + tx;
        if (b_row < M && b_col < K) {
            Bs[ty][tx] = b[b_row * K + b_col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Compute the partial dot product
        // We compute sum(As[ty][k] * Bs[tx][k])
        // As[ty][k] corresponds to A[row][t*32+k]
        // Bs[tx][k] corresponds to B[by*32+tx][t*32+k] which is B[col][t*32+k]
        // Note: Bs is accessed as Bs[tx][k]. 
        // Since Bs is padded, stride is 33. No bank conflict.
        for (int k = 0; k < TILE_SIZE; ++k) {
            c_val += As[ty][k] * Bs[tx][k];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    if (row < N && col < M) {
        // Apply bias and scale
        // Bias and Scale are vectors of size M, indexed by 'col'
        float b_val = bias[col];
        float s_val = scale[col];
        c[row * M + col] = (c_val + b_val) * s_val;
    }
}

torch::Tensor gemm_scale_bias_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor bias, torch::Tensor scale) {
    // Dimensions:
    // a: (N, K) -> Input
    // b: (M, K) -> Weight (transposed logic)
    // bias: (M)
    // scale: (M)
    // output: (N, M)
    
    int N = a.size(0);
    int K = a.size(1);
    int M = b.size(0);

    auto c = torch::zeros({N, M}, a.options());

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_scale_bias_kernel<<<dimGrid, dimBlock>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        scale.data_ptr<float>(), 
        c.data_ptr<float>(), 
        N, M, K
    );

    return c;
}
"""

fused_gemm_scale_cpp_source = (
    "torch::Tensor gemm_scale_bias_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor bias, torch::Tensor scale);"
)

# Compile the inline CUDA code
fused_gemm_scale_op = load_inline(
    name="fused_gemm_scale",
    cpp_sources=fused_gemm_scale_cpp_source,
    cuda_sources=fused_gemm_scale_source,
    functions=["gemm_scale_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused GEMM, Bias, and Scaling,
    followed by the standard PyTorch BatchNorm1d.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Use nn.Linear to ensure correct initialization and state_dict compatibility
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        
        # Load the custom operator
        self.fused_op = fused_gemm_scale_op

    def forward(self, x):
        # Use the custom fused kernel for Linear + Scale
        # self.gemm.weight is (out_features, in_features) -> (M, K)
        # self.gemm.bias is (out_features) -> (M)
        # self.scale is (out_features) -> (M)
        x = self.fused_op.gemm_scale_bias_cuda(x, self.gemm.weight, self.gemm.bias, self.scale)
        
        # Apply BatchNorm
        x = self.bn(x)
        return x