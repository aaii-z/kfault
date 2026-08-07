import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, Bias, and Scaling
# This kernel computes: Output = (Input * Weight^T + Bias) * Scale
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
    // We map threads to output elements. 
    // Row index in C (and A)
    int row = bx * TILE_SIZE + ty;
    // Col index in C (and B, Scale, Bias)
    int col = by * TILE_SIZE + tx;

    // Accumulator for the dot product
    float c_val = 0.0f;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over all tiles required to compute the element C[row][col]
    // A is (N, K), B is (M, K) (we treat B as transposed for the multiplication A * B^T)
    // We iterate over the K dimension in steps of TILE_SIZE
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load a tile of A into shared memory
        // A is row-major: A[row][k]
        int a_col = t * TILE_SIZE + tx;
        if (row < N && a_col < K) {
            As[ty][tx] = a[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load a tile of B into shared memory
        // We need B^T, so we need elements of B[col][k]
        // B is row-major: B[col][k]
        // We are loading a tile that corresponds to B's rows (indexed by 'col') and columns (indexed by 'k')
        // To coalesce global memory access for B, we want adjacent threads (tx) to access adjacent memory.
        // However, here adjacent threads (tx) correspond to adjacent 'col's.
        // B[col][k] accesses are strided by K if we fix k and vary col.
        // But we are loading B[col][t*TILE_SIZE + ty].
        // Thread (tx, ty) loads B[by*TILE_SIZE + tx][t*TILE_SIZE + ty].
        // In a warp, tx varies. So we access B[row_0][k_y], B[row_1][k_y], ...
        // This is strided access. 
        // Optimization: We can try to load transposed into shared memory or rely on cache.
        // For simplicity and correctness in this custom op, we proceed with this logic.
        
        int b_col = t * TILE_SIZE + ty; // The k-index in B
        if (col < M && b_col < K) {
            // B is (M, K), accessing B[col][b_col]
            Bs[tx][ty] = b[col * K + b_col];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Compute the partial dot product
        // We compute sum(As[ty][k] * Bs[tx][k])
        // As[ty][k] corresponds to A[row][k]
        // Bs[tx][k] corresponds to B[col][k]
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
        # Define parameters to replace nn.Linear
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # Keep BatchNorm as is, as requested to replace specific operators
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        
        # Load the custom operator
        self.fused_op = fused_gemm_scale_op

    def forward(self, x):
        # Use the custom fused kernel for Linear + Scale
        # x: (Batch, In_Features)
        # weight: (Out_Features, In_Features)
        # bias: (Out_Features)
        # scale: (Out_Features)
        x = self.fused_op.gemm_scale_bias_cuda(x, self.weight, self.bias, self.scale)
        
        # Apply BatchNorm
        x = self.bn(x)
        return x