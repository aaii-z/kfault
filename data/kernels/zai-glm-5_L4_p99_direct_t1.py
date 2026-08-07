import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear + GELU + Softmax
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants for GELU approximation
const float GELU_CONST = 0.7978845608028654f; // sqrt(2/pi)
const float GELU_COEFF = 0.044715f;

// Kernel 1: Fused Linear + GELU
// Computes Y = GELU(X * W^T + bias)
template <int TILE_SIZE>
__global__ void linear_gelu_kernel(
    const float* __restrict__ X, 
    const float* __restrict__ W, 
    const float* __restrict__ bias, 
    float* __restrict__ Y, 
    int M, int N, int K) {
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    __shared__ float X_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float W_tile[TILE_SIZE][TILE_SIZE];
    
    float acc = 0.0f;
    
    // Loop over K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load X tile: X_tile[ty][tx] = X[row, k]
        int x_k = t * TILE_SIZE + tx;
        if (row < M && x_k < K) {
            X_tile[ty][tx] = X[row * K + x_k];
        } else {
            X_tile[ty][tx] = 0.0f;
        }
        
        // Load W tile: W_tile[ty][tx] = W[col, k]
        // We want W_tile[k][tx] in computation, so we load transposed into shared memory
        // or simply load W[col, k] into W_tile[ty][tx] and access it correctly.
        // To avoid bank conflicts in computation, we load W[col, k] into W_tile[ty][tx].
        // In computation: acc += X_tile[ty][k] * W_tile[k][tx].
        // So we need to load W such that W_tile[k_idx][tx] = W[col, k_idx].
        // Here ty corresponds to k_idx in loading loop.
        int w_k = t * TILE_SIZE + ty;
        if (col < N && w_k < K) {
            W_tile[ty][tx] = W[col * K + w_k];
        } else {
            W_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += X_tile[ty][k] * W_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        // Add bias
        if (bias != nullptr) {
            acc += bias[col];
        }
        
        // Apply GELU
        float val = acc;
        float inner = GELU_CONST * (val + GELU_COEFF * val * val * val);
        float tanh_out = tanhf(inner);
        float gelu_out = 0.5f * val * (1.0f + tanh_out);
        
        Y[row * N + col] = gelu_out;
    }
}

// Kernel 2: Row-wise Softmax
// One block per row
__global__ void softmax_kernel(const float* __restrict__ X, float* __restrict__ Y, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Pointers to the start of the row
    const float* row_x = X + row * N;
    float* row_y = Y + row * N;
    
    // 1. Find Max
    float thread_max = -INFINITY;
    for (int i = tid; i < N; i += block_size) {
        thread_max = fmaxf(thread_max, row_x[i]);
    }
    
    // Shared memory reduction for max
    __shared__ float s_max[256]; // Assuming block_size <= 256
    s_max[tid] = thread_max;
    __syncthreads();
    
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float row_max = s_max[0];
    
    // 2. Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += block_size) {
        thread_sum += expf(row_x[i] - row_max);
    }
    
    __shared__ float s_sum[256];
    s_sum[tid] = thread_sum;
    __syncthreads();
    
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = s_sum[0];
    
    // 3. Normalize and write output
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < N; i += block_size) {
        row_y[i] = expf(row_x[i] - row_max) * inv_sum;
    }
}

// Wrapper function
torch::Tensor fused_linear_gelu_softmax_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);
    
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto y = torch::empty({M, N}, options);
    
    // Launch Linear + GELU kernel
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    linear_gelu_kernel<TILE_SIZE><<<blocks, threads>>>(
        x.data_ptr<float>(), 
        w.data_ptr<float>(), 
        b.defined() && b.numel() > 0 ? b.data_ptr<float>() : nullptr, 
        y.data_ptr<float>(), 
        M, N, K
    );
    
    // Launch Softmax kernel
    // One block per row. Block size 256 is reasonable.
    int block_size = 256;
    int num_blocks = M;
    
    softmax_kernel<<<num_blocks, block_size>>>(y.data_ptr<float>(), y.data_ptr<float>(), M, N);
    
    return y;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_linear_gelu_softmax_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_linear_gelu_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Define weights and bias manually to match nn.Linear
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize weights (mimic nn.Linear initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        self.fused_op = fused_ops

    def forward(self, x):
        # Call the custom fused kernel
        return self.fused_op.fused_linear_gelu_softmax_cuda(x, self.weight, self.bias)