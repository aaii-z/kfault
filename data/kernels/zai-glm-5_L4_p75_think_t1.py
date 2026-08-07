import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm, Min, and Bias Add
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_gn_min_bias_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ gn_weight, 
    const float* __restrict__ gn_bias, 
    const float* __restrict__ bias_add, 
    float* __restrict__ output, 
    int batch_size, 
    int channels, 
    int num_groups, 
    float eps) {
    
    int b = blockIdx.x;
    int g = threadIdx.x;
    
    if (b >= batch_size) return;
    
    int group_size = channels / num_groups;
    if (group_size == 0) return;
    
    // Shared memory for reduction (max 1024 threads per block)
    extern __shared__ float s_min[];
    
    float local_min = INFINITY;
    
    if (g < num_groups) {
        int start_idx = g * group_size;
        const float* x_ptr = input + b * channels + start_idx;
        const float* w_ptr = gn_weight + start_idx;
        const float* b_ptr = gn_bias + start_idx;
        
        // Compute mean and var for the group
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int i = 0; i < group_size; ++i) {
            float val = x_ptr[i];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / group_size;
        float var = sum_sq / group_size - mean * mean;
        float inv_std = rsqrtf(var + eps);
        
        // Find min in normalized group
        for (int i = 0; i < group_size; ++i) {
            float val = x_ptr[i];
            float norm_val = (val - mean) * inv_std * w_ptr[i] + b_ptr[i];
            if (norm_val < local_min) {
                local_min = norm_val;
            }
        }
        s_min[g] = local_min;
    }
    
    __syncthreads();
    
    // Parallel reduction to find global min for this batch item
    for (int s = num_groups / 2; s > 0; s >>= 1) {
        if (g < s) {
            if (s_min[g + s] < s_min[g]) {
                s_min[g] = s_min[g + s];
            }
        }
        __syncthreads();
    }
    
    // s_min[0] now contains the global minimum for batch b
    float global_min = s_min[0];
    
    // Write output
    // Each thread writes its group's portion of the output
    if (g < num_groups) {
        int start_idx = g * group_size;
        float* out_ptr = output + b * channels + start_idx;
        const float* bias_ptr = bias_add + start_idx;
        
        for (int i = 0; i < group_size; ++i) {
            out_ptr[i] = global_min + bias_ptr[i];
        }
    }
}

torch::Tensor fused_gn_min_bias_cuda(
    torch::Tensor input, 
    torch::Tensor gn_weight, 
    torch::Tensor gn_bias, 
    torch::Tensor bias_add, 
    int num_groups) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    float eps = 1e-5;
    
    // Output shape (B, C, 1, 1)
    auto output = torch::empty({batch_size, channels, 1, 1}, input.options());
    
    int threads = num_groups;
    int shared_mem_size = threads * sizeof(float);
    
    fused_gn_min_bias_kernel<<<batch_size, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        bias_add.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        num_groups,
        eps
    );
    
    return output;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_gn_min_bias_cuda(torch::Tensor input, torch::Tensor gn_weight, torch::Tensor gn_bias, torch::Tensor bias_add, int num_groups);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gn_min_bias",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gn_min_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a GEMM, and a fused Group Normalization + Minimum + Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups
        self.fused_op = fused_op

    def forward(self, x):
        x = self.gemm(x)
        # Fused GroupNorm + Min + Bias
        # We pass the group_norm weight and bias, and the external bias
        x = self.fused_op.fused_gn_min_bias_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.bias.view(-1), 
            self.num_groups
        )
        return x