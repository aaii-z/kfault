import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM + GroupNorm + Swish + Multiply + Swish CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

constexpr int TILE_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_gemm_groupnorm_swish_multiply_swish_kernel(
    const float* input, const float* weight, const float* bias, 
    const float* multiply_weight, float* output,
    int batch_size, int in_features, int out_features, int num_groups) {
    
    extern __shared__ float shared_mem[];
    
    int group_size = out_features / num_groups;
    int group_id = blockIdx.y;
    int group_start = group_id * group_size;
    int group_end = group_start + group_size;
    
    int local_idx = threadIdx.x;
    int global_idx = group_start + local_idx;
    
    if (global_idx >= out_features) return;
    
    // Compute GEMM for this thread's output element
    float sum = 0.0f;
    for (int i = 0; i < in_features; i += TILE_SIZE) {
        if (i + local_idx < in_features) {
            float input_val = input[blockIdx.x * in_features + i + local_idx];
            float weight_val = weight[global_idx * in_features + i + local_idx];
            sum += input_val * weight_val;
        }
    }
    
    if (bias != nullptr) {
        sum += bias[global_idx];
    }
    
    // Store in shared memory for GroupNorm reduction
    shared_mem[local_idx] = sum;
    __syncthreads();
    
    // GroupNorm: compute mean and variance for this group
    float group_sum = 0.0f;
    float group_sq_sum = 0.0f;
    int valid_threads = min(group_size, out_features - group_start);
    
    for (int i = 0; i < valid_threads; i++) {
        group_sum += shared_mem[i];
        group_sq_sum += shared_mem[i] * shared_mem[i];
    }
    
    float mean = group_sum / valid_threads;
    float variance = (group_sq_sum / valid_threads) - (mean * mean);
    float inv_std = 1.0f / sqrtf(variance + 1e-5f);
    
    // Apply GroupNorm and Swish
    float normalized = (shared_mem[local_idx] - mean) * inv_std;
    float swished = normalized * sigmoid(normalized);
    
    // Apply multiply weight and second Swish
    float multiplied = swished * multiply_weight[global_idx];
    float final_output = multiplied * sigmoid(multiplied);
    
    output[blockIdx.x * out_features + global_idx] = final_output;
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor multiply_weight, int num_groups) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 grid(batch_size, num_groups);
    int group_size = out_features / num_groups;
    int shared_mem_size = group_size * sizeof(float);
    
    fused_gemm_groupnorm_swish_multiply_swish_kernel<<<grid, group_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        multiply_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_features, out_features, num_groups);
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_operation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor multiply_weight, int num_groups);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Store weight and bias as parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_operation_cuda(
            x, self.weight, self.bias, self.multiply_weight, self.num_groups
        )