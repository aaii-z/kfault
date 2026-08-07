import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM + GroupNorm + Min + Bias CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__device__ float warp_reduce_min(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void fused_gemm_groupnorm_min_bias_kernel(
    const float* input, const float* weight, const float* bias, 
    float* output, int M, int N, int K, int num_groups, int group_size) {
    
    extern __shared__ float shared_mem[];
    float* row_shared = shared_mem;
    float* col_shared = shared_mem + blockDim.x;
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k += TILE_DIM) {
            if (threadIdx.y < TILE_DIM && k + threadIdx.y < K) {
                row_shared[threadIdx.x * TILE_DIM + threadIdx.y] = 
                    input[row * K + k + threadIdx.y];
            }
            if (threadIdx.x < TILE_DIM && k + threadIdx.x < K) {
                col_shared[threadIdx.y * TILE_DIM + threadIdx.x] = 
                    weight[(k + threadIdx.x) * N + col];
            }
            __syncthreads();
            
            int tile_end = min(TILE_DIM, K - k);
            for (int i = 0; i < tile_end; i++) {
                sum += row_shared[threadIdx.x * TILE_DIM + i] * 
                       col_shared[threadIdx.y * TILE_DIM + i];
            }
            __syncthreads();
        }
        
        output[row * N + col] = sum;
    }
    
    __syncthreads();
    
    // Group Normalization and Min reduction
    if (col < N && row < M) {
        int group_id = col / group_size;
        int group_start = group_id * group_size;
        int group_end = min((group_id + 1) * group_size, N);
        
        // Compute group mean and variance
        float group_sum = 0.0f;
        float group_sq_sum = 0.0f;
        
        for (int i = group_start; i < group_end; i++) {
            float val = output[row * N + i];
            group_sum += val;
            group_sq_sum += val * val;
        }
        
        float group_mean = group_sum / group_size;
        float group_var = (group_sq_sum / group_size) - (group_mean * group_mean);
        float inv_std = rsqrtf(group_var + 1e-5f);
        
        // Normalize and find min
        float min_val = FLT_MAX;
        for (int i = group_start; i < group_end; i++) {
            float normalized = (output[row * N + i] - group_mean) * inv_std;
            output[row * N + i] = normalized;
            min_val = fminf(min_val, normalized);
        }
        
        // Apply min reduction and bias
        if (threadIdx.x == 0) {
            output[row * N + col] = min_val + bias[col];
        }
    }
}

torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor weight, 
                                  torch::Tensor bias, int num_groups) {
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    int group_size = N / num_groups;
    
    auto output = torch::empty({M, N}, input.options());
    
    dim3 block_dim(32, 32);
    dim3 grid_dim((M + block_dim.x - 1) / block_dim.x, 
                  (N + block_dim.y - 1) / block_dim.y);
    
    int shared_mem_size = (block_dim.x * TILE_DIM + block_dim.y * TILE_DIM) * sizeof(float);
    
    fused_gemm_groupnorm_min_bias_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), M, N, K, num_groups, group_size);
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor weight, 
                                  torch::Tensor bias, int num_groups);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape).squeeze().unsqueeze(0))
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_operation_cuda(x, self.weight, self.bias, self.num_groups)