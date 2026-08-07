import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear-sigmoid-sum
# Optimization Strategy:
# 1. Use Tensor Cores via WMMA (Warp Matrix Multiply Accumulate) for the Linear layer (FP32 inputs/accumulation).
# 2. Perform the reduction (Sum) efficiently using Shared Memory and Warp Shuffles.
# 3. Fuse Sigmoid and Sum to save memory bandwidth.

fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;

// Block size configuration
// We use a 2D grid of blocks: (Batch, Hidden / Block_H)
// Inside a block, we use a 2D grid of warps to compute tiles of the output matrix.
// Each warp computes a 16x16 tile of the matrix multiplication.

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Optimized Block Dimensions for 1024 threads (32 warps)
constexpr int BLOCK_H = 64;  
constexpr int BLOCK_W = 128; 
constexpr int WARP_COUNT_H = BLOCK_H / WMMA_M; // 4
constexpr int WARP_COUNT_W = BLOCK_W / WMMA_N; // 8
constexpr int WARP_COUNT = WARP_COUNT_H * WARP_COUNT_W; // 32

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_linear_sigmoid_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size) {

    // Shared memory for the tile of matrix multiplication results
    __shared__ float s_tile[BLOCK_H][BLOCK_W];
    // Shared memory for partial sums during reduction
    __shared__ float s_partial_sums[BLOCK_W][WARP_COUNT_H + 1]; 

    int batch_block_idx = blockIdx.x;
    int hidden_block_idx = blockIdx.y;

    int batch_start = batch_block_idx * BLOCK_W;
    int hidden_start = hidden_block_idx * BLOCK_H;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int warp_row = warp_id / WARP_COUNT_W; 
    int warp_col = warp_id % WARP_COUNT_W; 

    int hidden_warp_idx = hidden_start + warp_row * WMMA_M;
    int batch_warp_idx = batch_start + warp_col * WMMA_N;

    // Initialize accumulator for Matrix Multiply
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the input dimension (K dimension) in steps of WMMA_K
    for (int k = 0; k < input_size; k += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float, wmma::col_major> b_frag;

        if (hidden_warp_idx < hidden_size && batch_warp_idx < batch_size) {
            // Load Weight tile (Matrix A)
            wmma::load_matrix_sync(a_frag, weight + hidden_warp_idx * input_size + k, input_size);
            
            // Load Input tile (Matrix B)
            // We treat Input as the B matrix in C = A * B.
            // We want C = W * X^T.
            // A = W (16x16)
            // B = X^T (16x16). X is (Batch, Input). X^T is (Input, Batch).
            // We load X with layout col_major to effectively load X^T.
            wmma::load_matrix_sync(b_frag, input + batch_warp_idx * input_size + k, input_size);
            
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store the accumulator to shared memory
    int store_row = warp_row * WMMA_M;
    int store_col = warp_col * WMMA_N;
    
    if (hidden_warp_idx < hidden_size && batch_warp_idx < batch_size) {
        wmma::store_matrix_sync(&s_tile[store_row][store_col], acc_frag, BLOCK_W, wmma::mem_row_major);
    }

    __syncthreads();

    // Reduction and Activation
    int tid = threadIdx.x;
    int col_in_block = tid % BLOCK_W;
    int row_group = tid / BLOCK_W; // 0 to 7 (since 1024 / 128 = 8 groups)
    
    // Each thread processes BLOCK_H / 8 = 8 rows
    float local_sum = 0.0f;
    
    for (int r = 0; r < 8; ++r) {
        int row_in_block = row_group * 8 + r;
        int hidden_idx = hidden_start + row_in_block;
        int batch_idx = batch_start + col_in_block;
        
        if (hidden_idx < hidden_size && batch_idx < batch_size) {
            float val = s_tile[row_in_block][col_in_block];
            // Add bias
            val += bias[hidden_idx];
            // Sigmoid
            val = sigmoid(val);
            local_sum += val;
        }
    }
    
    s_partial_sums[col_in_block][row_group] = local_sum;
    
    __syncthreads();
    
    // Final reduction
    if (tid < BLOCK_W) {
        float col_sum = 0.0f;
        // Sum 8 partials
        for (int i = 0; i < 8; ++i) {
            col_sum += s_partial_sums[tid][i];
        }
        atomicAdd(&output[batch_start + tid], col_sum);
    }
}

torch::Tensor fused_linear_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);

    // Output tensor initialized to zero
    auto output = torch::zeros({batch_size, 1}, input.options());

    // Grid configuration
    dim3 gridDim(
        (batch_size + BLOCK_W - 1) / BLOCK_W,
        (hidden_size + BLOCK_H - 1) / BLOCK_H
    );
    
    // Launch kernel
    fused_linear_sigmoid_sum_kernel<<<gridDim, 1024>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return output;
}
"""

fused_linear_sigmoid_sum_cpp_source = (
    "torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_linear_sigmoid_sum_v3",
    cpp_sources=fused_linear_sigmoid_sum_cpp_source,
    cuda_sources=fused_linear_sigmoid_sum_source,
    functions=["fused_linear_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused Linear-Sigmoid-Sum.
    Uses WMMA (Tensor Cores) for high throughput.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_linear_sigmoid_sum_cuda(x, self.weight, self.bias)