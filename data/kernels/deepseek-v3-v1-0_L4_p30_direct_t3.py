import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused GEMM + GroupNorm + HardTanh CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_gemm_groupnorm_hardtanh_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    int num_groups, float min_val, float max_val, float eps) {
    
    const int group_size = out_features / num_groups;
    const int group_id = blockIdx.x % num_groups;
    const int batch_id = blockIdx.x / num_groups;
    const int feature_id = threadIdx.x;
    
    if (batch_id >= batch_size || feature_id >= group_size) return;
    
    const int output_idx = batch_id * out_features + group_id * group_size + feature_id;
    const int input_base = batch_id * in_features;
    
    // Compute GEMM
    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        int weight_idx = (group_id * group_size + feature_id) * in_features + i;
        sum += input[input_base + i] * weight[weight_idx];
    }
    sum += bias[group_id * group_size + feature_id];
    
    // Store intermediate result for group norm computation
    __shared__ float shared_sums[1024];
    __shared__ float shared_sq_sums[1024];
    __shared__ float mean, variance;
    
    shared_sums[threadIdx.x] = sum;
    shared_sq_sums[threadIdx.x] = sum * sum;
    __syncthreads();
    
    // Reduce within group for mean and variance
    if (threadIdx.x == 0) {
        float group_sum = 0.0f;
        float group_sq_sum = 0.0f;
        for (int i = 0; i < group_size; ++i) {
            group_sum += shared_sums[i];
            group_sq_sum += shared_sq_sums[i];
        }
        mean = group_sum / group_size;
        variance = group_sq_sum / group_size - mean * mean;
    }
    __syncthreads();
    
    // Apply group normalization
    float normalized = (sum - mean) / sqrtf(variance + eps);
    
    // Apply HardTanh
    float result = normalized;
    result = fmaxf(min_val, result);
    result = fminf(max_val, result);
    
    output[output_idx] = result;
}

torch::Tensor fused_gemm_groupnorm_hardtanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int num_groups, float min_val, float max_val, float eps=1e-5) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    auto group_size = out_features / num_groups;
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 grid_dim(batch_size * num_groups);
    dim3 block_dim(group_size);
    
    fused_gemm_groupnorm_hardtanh_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        min_val,
        max_val,
        eps
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_gemm_groupnorm_hardtanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int num_groups, float min_val, float max_val, float eps=1e-5);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_groupnorm_hardtanh",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_groupnorm_hardtanh_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs fused GEMM + GroupNorm + HardTanh operation.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Initialize weights and bias for GEMM
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Store the fused operation
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.fused_op.fused_gemm_groupnorm_hardtanh_cuda(
            x, self.weight, self.bias, 
            self.num_groups, self.hardtanh_min, self.hardtanh_max
        )