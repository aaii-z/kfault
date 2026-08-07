import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants for tiling
const int TILE_K = 32;
const int TILE_N = 64;
const int BATCH_TILE = 8;

// Kernel 1: Fused Linear + Sigmoid
// Computes C = sigmoid(A * W^T + bias)
// A: (M, K), W: (N, K), bias: (N), C: (M, N)
// Grid: (N/TILE_N, M/BATCH_TILE), Block: (TILE_N, BATCH_TILE)
__global__ void linear_sigmoid_kernel(const float* A, const float* W, const float* bias, float* C, int M, int K, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int col = blockIdx.x * TILE_N + tx; // N dimension
    int row = blockIdx.y * BATCH_TILE + ty; // M dimension
    
    __shared__ float s_A[BATCH_TILE][TILE_K];
    __shared__ float s_W[TILE_N][TILE_K];
    
    float sum = 0.0f;
    
    // Loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load tile from A into shared memory
        // Threads cooperatively load BATCH_TILE * TILE_K elements
        int a_idx = row * K + k_tile + tx;
        if (row < M && (k_tile + tx) < K) {
            s_A[ty][tx] = A[a_idx];
        } else {
            s_A[ty][tx] = 0.0f;
        }
        
        // Load tile from W into shared memory
        // W is (N, K), we need W[col, k]
        // Threads cooperatively load TILE_N * TILE_K elements
        int w_idx = col * K + k_tile + ty;
        if (col < N && (k_tile + ty) < K) {
            s_W[tx][ty] = W[w_idx];
        } else {
            s_W[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_K; ++k) {
            sum += s_A[ty][k] * s_W[tx][k];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        sum += bias[col];
        // Sigmoid activation
        C[row * N + col] = 1.0f / (1.0f + expf(-sum));
    }
}

// Kernel 2: Fused Linear + LogSumExp
// Computes out = logsumexp(A * W^T + bias, dim=1)
// A: (M, K), W: (N, K), bias: (N), out: (M)
// Grid: (M/BATCH_TILE), Block: (BATCH_TILE * TILE_N)
__global__ void linear_logsumexp_kernel(const float* A, const float* W, const float* bias, float* out, int M, int K, int N) {
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    int row_in_tile = tid / TILE_N;
    int col_in_tile = tid % TILE_N;
    
    int row = batch_idx * BATCH_TILE + row_in_tile;
    
    // Shared memory for input tile and weights tile
    __shared__ float s_A[BATCH_TILE][TILE_K];
    __shared__ float s_W[TILE_N][TILE_K];
    
    // Running max and sum for online logsumexp
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    
    // Loop over N dimension in tiles (output columns)
    for (int n_tile = 0; n_tile < N; n_tile += TILE_N) {
        int col = n_tile + col_in_tile;
        
        float sum = 0.0f;
        
        // Loop over K dimension
        for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
            // Load A tile
            int a_idx = row * K + k_tile + col_in_tile;
            if (row < M && (k_tile + col_in_tile) < K) {
                s_A[row_in_tile][col_in_tile] = A[a_idx];
            } else {
                s_A[row_in_tile][col_in_tile] = 0.0f;
            }
            
            // Load W tile
            // W is (N, K), we need W[col, k]
            int w_idx = col * K + k_tile + row_in_tile;
            if (col < N && (k_tile + row_in_tile) < K) {
                s_W[col_in_tile][row_in_tile] = W[w_idx];
            } else {
                s_W[col_in_tile][row_in_tile] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot product
            // Each thread computes one element of the output tile
            for (int k = 0; k < TILE_K; ++k) {
                sum += s_A[row_in_tile][k] * s_W[col_in_tile][k];
            }
            __syncthreads();
        }
        
        if (row < M && col < N) {
            sum += bias[col];
            
            // Online reduction for LogSumExp
            // We need to reduce across the N dimension (within the tile)
            // Use shared memory for reduction within the tile
            
            // First, find max and sum(exp(x-max)) for this thread's value
            // But we need to combine with previous tiles.
            // We can just accumulate (max, sum) pairs.
            
            // Local value
            float val = sum;
            
            // Reduce within the tile (TILE_N threads per row)
            // We need to reduce values belonging to the same row.
            // Threads with same row_in_tile are in the same row.
            
            // Shared memory for tile reduction
            __shared__ float s_tile_max[BATCH_TILE][TILE_N]; // Max for each row in tile
            __shared__ float s_tile_sum[BATCH_TILE][TILE_N]; // Sum for each row in tile
            
            s_tile_max[row_in_tile][col_in_tile] = val;
            s_tile_sum[row_in_tile][col_in_tile] = 1.0f; // exp(0)
            
            // Parallel reduction to find tile max
            for (int s = TILE_N / 2; s > 0; s >>= 1) {
                if (col_in_tile < s) {
                    float m1 = s_tile_max[row_in_tile][col_in_tile];
                    float m2 = s_tile_max[row_in_tile][col_in_tile + s];
                    if (m2 > m1) {
                        s_tile_max[row_in_tile][col_in_tile] = m2;
                        s_tile_sum[row_in_tile][col_in_tile] = s_tile_sum[row_in_tile][col_in_tile] * expf(m1 - m2) + s_tile_sum[row_in_tile][col_in_tile + s];
                    } else {
                        s_tile_max[row_in_tile][col_in_tile] = m1;
                        s_tile_sum[row_in_tile][col_in_tile] = s_tile_sum[row_in_tile][col_in_tile] + s_tile_sum[row_in_tile][col_in_tile + s] * expf(m2 - m1);
                    }
                }
                __syncthreads();
            }
            
            // Now thread 0 of the row group has the tile's (max, sum)
            if (col_in_tile == 0) {
                float tile_max = s_tile_max[row_in_tile][0];
                float tile_sum = s_tile_sum[row_in_tile][0];
                
                // Merge with running total
                float new_max = max(running_max, tile_max);
                if (new_max > -INFINITY) {
                    if (running_max > -INFINITY) {
                         running_sum = running_sum * expf(running_max - new_max) + tile_sum * expf(tile_max - new_max);
                    } else {
                         running_sum = tile_sum * expf(tile_max - new_max);
                    }
                }
                running_max = new_max;
            }
        }
        __syncthreads(); // Ensure all threads are done before next tile
    }
    
    // Write output
    if (row < M && col_in_tile == 0) {
        out[row] = logf(running_sum) + running_max;
    }
}

// C++ wrapper functions
torch::Tensor linear_sigmoid_cuda(torch::Tensor A, torch::Tensor W, torch::Tensor bias) {
    int M = A.size(0);
    int K = A.size(1);
    int N = W.size(0);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 block(TILE_N, BATCH_TILE);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + BATCH_TILE - 1) / BATCH_TILE);
    
    linear_sigmoid_kernel<<<grid, block>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), bias.data_ptr<float>(), C.data_ptr<float>(), M, K, N
    );
    
    return C;
}

torch::Tensor linear_logsumexp_cuda(torch::Tensor A, torch::Tensor W, torch::Tensor bias) {
    int M = A.size(0);
    int K = A.size(1);
    int N = W.size(0);
    
    auto out = torch::zeros({M}, A.options());
    
    const int block_size = BATCH_TILE * TILE_N;
    dim3 grid(M / BATCH_TILE); // Assuming M is divisible
    
    linear_logsumexp_kernel<<<grid, block_size>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), M, K, N
    );
    
    return out;
}
"""

cpp_source = """
torch::Tensor linear_sigmoid_cuda(torch::Tensor A, torch::Tensor W, torch::Tensor bias);
torch::Tensor linear_logsumexp_cuda(torch::Tensor A, torch::Tensor W, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["linear_sigmoid_cuda", "linear_logsumexp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        # Define weights and biases as parameters
        # We use nn.Parameter so they are recognized as model parameters
        # Linear 1: input_size -> hidden_size
        self.weight1 = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias1 = nn.Parameter(torch.randn(hidden_size))
        
        # Linear 2: hidden_size -> output_size
        self.weight2 = nn.Parameter(torch.randn(output_size, hidden_size))
        self.bias2 = nn.Parameter(torch.randn(output_size))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        # Fused Linear + Sigmoid
        x = self.fused_ops.linear_sigmoid_cuda(x, self.weight1, self.bias1)
        
        # Fused Linear + LogSumExp
        x = self.fused_ops.linear_logsumexp_cuda(x, self.weight2, self.bias2)
        
        return x