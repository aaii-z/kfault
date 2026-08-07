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
    
    // Shared memory for input data and reduction
    // Layout: [channels] for input data + [blockDim.x] for min reduction
    extern __shared__ float smem[];
    float* s_data = smem;
    float* s_min = smem + channels;
    
    // Load input to shared memory (coalesced)
    for (int i = g; i < channels; i += blockDim.x) {
        s_data[i] = input[b * channels + i];
    }
    __syncthreads();
    
    int group_size = channels / num_groups;
    float local_min = INFINITY;
    
    if (g < num_groups) {
        int start_idx = g * group_size;
        
        // Compute mean and var for the group
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int i = 0; i < group_size; ++i) {
            float val = s_data[start_idx + i];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / group_size;
        float var = sum_sq / group_size - mean * mean;
        float inv_std = rsqrtf(var + eps);
        
        // Find min in normalized group
        for (int i = 0; i < group_size; ++i) {
            float val = s_data[start_idx + i];
            // Apply GroupNorm: (x - mean) * inv_std * weight + bias
            float norm_val = (val - mean) * inv_std * gn_weight[start_idx + i] + gn_bias[start_idx + i];
            if (norm_val < local_min) {
                local_min = norm_val;
            }
        }
        s_min[g] = local_min;
    } else {
        s_min[g] = INFINITY;
    }
    
    __syncthreads();
    
    // Parallel reduction to find global min for this batch item
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (g < s) {
            if (s_min[g + s] < s_min[g]) {
                s_min[g] = s_min[g + s];
            }
        }
        __syncthreads();
    }
    
    float global_min = s_min[0];
    
    // Write output
    // Output shape: (1, C, B, 1)
    // Index calculation: c * B + b
    if (g < num_groups) {
        int start_idx = g * group_size;
        for (int i = 0; i < group_size; ++i) {
            int c = start_idx + i;
            output[c * batch_size + b] = global_min + bias_add[c];
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
    
    // Output shape (1, C, B, 1)
    auto output = torch::empty({1, channels, batch_size, 1}, input.options());
    
    int threads = num_groups;
    // Round up to power of 2 for reduction
    int pow2_threads = 1;
    while (pow2_threads < num_groups) pow2_threads <<= 1;
    
    // Shared memory size: channels (input) + pow2_threads (reduction)
    int shared_mem_size = (channels + pow2_threads) * sizeof(float);
    
    fused_gn_min_bias_kernel<<<batch_size, pow2_threads, shared_mem_size>>>(
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