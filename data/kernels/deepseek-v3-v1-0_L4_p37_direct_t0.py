import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + swish + bias
fused_matmul_swish_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

__device__ float swish_activation(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_matmul_swish_bias_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[k * out_features + col];
        }
        sum += bias[col];
        output[row * out_features + col] = swish_activation(sum);
    }
}

torch::Tensor fused_matmul_swish_bias_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(1);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );
    
    fused_matmul_swish_bias_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_matmul_swish_bias_cpp_source = "torch::Tensor fused_matmul_swish_bias_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Define optimized GroupNorm CUDA kernel
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void group_norm_forward_kernel(
    const float* input, float* output, const float* weight, const float* bias,
    int batch_size, int channels, int group_size, int elements_per_group,
    float eps) {
    
    int group_idx = blockIdx.x;
    int sample_idx = blockIdx.y;
    int local_idx = threadIdx.x;
    
    __shared__ float shared_mean;
    __shared__ float shared_variance;
    
    if (local_idx == 0) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        int start = (sample_idx * channels + group_idx * group_size) * elements_per_group;
        int end = start + group_size * elements_per_group;
        
        for (int i = start; i < end; i++) {
            float val = input[i];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / (group_size * elements_per_group);
        float variance = (sum_sq / (group_size * elements_per_group)) - (mean * mean);
        
        shared_mean = mean;
        shared_variance = 1.0f / sqrtf(variance + eps);
    }
    
    __syncthreads();
    
    int start_idx = (sample_idx * channels + group_idx * group_size) * elements_per_group;
    for (int i = local_idx; i < group_size * elements_per_group; i += blockDim.x) {
        int idx = start_idx + i;
        int channel_idx = group_idx * group_size + (i / elements_per_group);
        
        float normalized = (input[idx] - shared_mean) * shared_variance;
        output[idx] = normalized * weight[channel_idx] + bias[channel_idx];
    }
}

torch::Tensor group_norm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int num_groups, float eps=1e-5) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int group_size = channels / num_groups;
    int elements_per_group = 1;  // Assuming 1D features
    
    auto output = torch::empty_like(input);
    
    dim3 block_size(256);
    dim3 grid_size(num_groups, batch_size);
    
    group_norm_forward_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        channels,
        group_size,
        elements_per_group,
        eps
    );
    
    return output;
}
"""

group_norm_cpp_source = "torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps=1e-5);"

# Compile the inline CUDA code
fused_matmul_swish_bias = load_inline(
    name="fused_matmul_swish_bias",
    cpp_sources=fused_matmul_swish_bias_cpp_source,
    cuda_sources=fused_matmul_swish_bias_source,
    functions=["fused_matmul_swish_bias_cuda"],
    verbose=False
)

group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm_weight = nn.Parameter(torch.ones(out_features))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_features))
        self.num_groups = num_groups
        self.fused_op = fused_matmul_swish_bias
        self.group_norm_op = group_norm

    def forward(self, x):
        x = self.fused_op.fused_matmul_swish_bias_cuda(x, self.weight, self.bias)
        x = self.group_norm_op.group_norm_cuda(x, self.group_norm_weight, self.group_norm_bias, self.num_groups)
        return x