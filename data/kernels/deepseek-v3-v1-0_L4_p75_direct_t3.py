import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM + GroupNorm + Min + Bias CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__global__ void fused_gemm_groupnorm_min_bias_kernel(
    const float* input, const float* weight, const float* bias, 
    float* output, int M, int N, int K, int num_groups, int group_size, int bias_size) {
    
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    __shared__ float tile[32][33];
    
    // GEMM computation
    float sum = 0.0f;
    for (int t = 0; t < (K + 31) / 32; ++t) {
        int t_col = t * 32 + threadIdx.x;
        int t_row = t * 32 + threadIdx.y;
        
        if (row < M && t_col < K) {
            tile[threadIdx.y][threadIdx.x] = input[row * K + t_col];
        } else {
            tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (t_row < K && col < N) {
            tile[threadIdx.y][threadIdx.x + 16] = weight[t_row * N + col];
        } else {
            tile[threadIdx.y][threadIdx.x + 16] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < 32; ++i) {
            sum += tile[threadIdx.y][i] * tile[i][threadIdx.x + 16];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
    
    __syncthreads();
    
    // Group Normalization
    if (row < M) {
        int group_id = col / group_size;
        if (threadIdx.x == 0) {
            int group_start = group_id * group_size;
            int group_end = min((group_id + 1) * group_size, N);
            
            float group_sum = 0.0f;
            float group_sq_sum = 0.0f;
            
            for (int i = group_start; i < group_end; i++) {
                float val = output[row * N + i];
                group_sum += val;
                group_sq_sum += val * val;
            }
            
            float group_mean = group_sum / (group_end - group_start);
            float group_var = (group_sq_sum / (group_end - group_start)) - (group_mean * group_mean);
            float inv_std = rsqrtf(group_var + 1e-5f);
            
            for (int i = group_start; i < group_end; i++) {
                output[row * N + i] = (output[row * N + i] - group_mean) * inv_std;
            }
        }
    }
    
    __syncthreads();
    
    // Min reduction and bias addition
    if (row < M && col == 0) {
        float min_val = FLT_MAX;
        for (int i = 0; i < N; i++) {
            min_val = fminf(min_val, output[row * N + i]);
        }
        output[row] = min_val + bias[0];
    }
}

torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor weight, 
                                  torch::Tensor bias, int num_groups) {
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    int group_size = N / num_groups;
    int bias_size = bias.size(0);
    
    auto output = torch::empty({M, N}, input.options());
    
    dim3 block_dim(16, 16);
    dim3 grid_dim((M + block_dim.y - 1) / block_dim.y, 
                  (N + block_dim.x - 1) / block_dim.x);
    
    int shared_mem_size = sizeof(float) * 32 * 33;
    
    fused_gemm_groupnorm_min_bias_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), M, N, K, num_groups, group_size, bias_size);
    
    return output.narrow(1, 0, 1).view(M, 1);
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
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape).squeeze())
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_operation_cuda(x, self.weight, self.bias, self.num_groups)