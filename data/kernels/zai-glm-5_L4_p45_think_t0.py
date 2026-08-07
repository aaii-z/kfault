import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Fused Linear + Sigmoid
// Computes C = sigmoid(A * B + bias)
// A: (M, K), B: (K, N), bias: (N), C: (M, N)
// B is expected to be transposed (K, N)
__global__ void linear_sigmoid_kernel(const float* A, const float* B, const float* bias, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Tiling for better memory access patterns
        __shared__ float s_A[16][16];
        __shared__ float s_B[16][16];
        
        // Iterate over K dimension in tiles of size 16
        for (int k_tile = 0; k_tile < K; k_tile += 16) {
            // Load tile from A
            int k_idx_A = k_tile + threadIdx.x;
            if (row < M && k_idx_A < K)
                s_A[threadIdx.y][threadIdx.x] = A[row * K + k_idx_A];
            else
                s_A[threadIdx.y][threadIdx.x] = 0.0f;
            
            // Load tile from B (transposed)
            int k_idx_B = k_tile + threadIdx.y;
            if (k_idx_B < K && col < N)
                s_B[threadIdx.y][threadIdx.x] = B[k_idx_B * N + col];
            else
                s_B[threadIdx.y][threadIdx.x] = 0.0f;
            
            __syncthreads();
            
            // Compute partial dot product
            for (int k = 0; k < 16; ++k) {
                sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
            }
            __syncthreads();
        }
        
        sum += bias[col];
        // Sigmoid activation: 1 / (1 + exp(-x))
        C[row * N + col] = 1.0f / (1.0f + expf(-sum));
    }
}

// Kernel 2: Fused Linear + LogSumExp
// Computes out = logsumexp(A * B + bias, dim=1)
// A: (M, K), B: (K, N), bias: (N), out: (M)
// B is expected to be transposed (K, N)
__global__ void linear_logsumexp_kernel(const float* A, const float* B, const float* bias, float* out, int M, int K, int N) {
    int row = blockIdx.x; // One block per row
    int tid = threadIdx.x;
    
    // Shared memory for input row tile to reduce global memory access
    // We tile K to fit in shared memory if K is large
    const int TILE_K = 512;
    __shared__ float s_A[TILE_K];
    
    // Each thread handles 4 columns of the output matrix (N=1024, BlockSize=256)
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    // Loop over K in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperatively load input tile
        for (int i = tid; i < TILE_K; i += blockDim.x) {
            int k_idx = k_tile + i;
            if (k_idx < K)
                s_A[i] = A[row * K + k_idx];
            else
                s_A[i] = 0.0f;
        }
        __syncthreads();
        
        // Compute partial dot products
        for (int k = 0; k < TILE_K; ++k) {
            float val_a = s_A[k];
            int k_global = k_tile + k;
            if (k_global < K) {
                int col_start = tid * 4;
                // Vectorized load for weights (float4)
                // Assuming N is divisible by 4 or bounds are handled
                if (col_start < N) {
                    float4 w = *reinterpret_cast<const float4*>(&B[k_global * N + col_start]);
                    sum0 += val_a * w.x;
                    sum1 += val_a * w.y;
                    sum2 += val_a * w.z;
                    sum3 += val_a * w.w;
                }
            }
        }
        __syncthreads();
    }
    
    // Add bias
    int col_start = tid * 4;
    if (col_start < N) {
        sum0 += bias[col_start];
        if(col_start+1 < N) sum1 += bias[col_start+1];
        if(col_start+2 < N) sum2 += bias[col_start+2];
        if(col_start+3 < N) sum3 += bias[col_start+3];
    }
    
    // Local LogSumExp reduction for the 4 values handled by this thread
    float vals[4] = {sum0, sum1, sum2, sum3};
    float local_max = max(max(vals[0], vals[1]), max(vals[2], vals[3]));
    float local_sum = 0.0f;
    
    // Handle case where all values might be -inf
    if (local_max > -INFINITY) {
        local_sum += expf(vals[0] - local_max);
        local_sum += expf(vals[1] - local_max);
        local_sum += expf(vals[2] - local_max);
        local_sum += expf(vals[3] - local_max);
    }
    
    // Block reduction to find global max and sum
    __shared__ float s_max[256];
    __shared__ float s_sum[256];
    
    s_max[tid] = local_max;
    s_sum[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction (can be optimized with warp shuffles, but simple loop is clear)
    if (tid == 0) {
        float global_max = s_max[0];
        float global_sum = 0.0f;
        
        // Find global max
        for (int i = 1; i < blockDim.x; ++i) {
            if (s_max[i] > global_max) global_max = s_max[i];
        }
        
        // Compute global sum scaled by global max
        if (global_max > -INFINITY) {
            for (int i = 0; i < blockDim.x; ++i) {
                // s_sum[i] is sum(exp(val - local_max))
                // We need sum(exp(val - global_max)) = sum(exp(val - local_max) * exp(local_max - global_max))
                global_sum += s_sum[i] * expf(s_max[i] - global_max);
            }
        }
        
        out[row] = logf(global_sum) + global_max;
    }
}

// C++ wrapper functions
torch::Tensor linear_sigmoid_cuda(torch::Tensor A