import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + swish + bias
fused_matmul_swish_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float swish_activation(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
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

__global__ void group_norm_forward_kernel(
    const float* input, float* output, const float* gamma, const float* beta,
    int batch_size, int num_channels, int num_groups, int channels_per_group,
    int elements_per_channel, float eps) {
    
    extern __shared__ float shared_data[];
    float* shared_mean = shared_data;
    float* shared_var = shared_data + 1;
    
    int group_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    
    int channel_start = group_idx * channels_per_group;
    int channel_end = min(channel_start + channels_per_group, num_channels);
    
    int input_offset = batch_idx * num_channels * elements_per_channel + channel_start * elements_per_channel;
    
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int count = 0;
        
        for (int c = 0; c < channels_per_group; c++) {
            int channel_idx = channel_start + c;
            if (channel_idx < num_channels) {
                for (int i = 0; i < elements_per_channel; i++) {
                    int idx = input_offset + c * elements_per_channel + i;
                    float val = input[idx];
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }
        }
        
        float mean = sum / count;
        float variance = (sum_sq / count) - (mean * mean);
        
        *shared_mean = mean;
        *shared_var = rsqrtf(variance + eps);
    }
    
    __syncthreads();
    
    float mean_val = *shared_mean;
    float inv_std_val = *shared_var;
    
    for (int c = threadIdx.x; c < channels_per_group; c += blockDim.x) {
        int channel_idx = channel_start + c;
        if (channel_idx < num_channels) {
            for (int i = 0; i < elements_per_channel; i++) {
                int idx = input_offset + c * elements_per_channel + i;
                float normalized = (input[idx] - mean_val) * inv_std_val;
                output[idx] = normalized * gamma[channel_idx] + beta[channel_idx];
            }
        }
    }
}

torch::Tensor group_norm_cuda(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor beta,
    int num_groups, float eps=1e-5) {
    
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int channels_per_group = num_channels / num_groups;
    int elements_per_channel = 1;  // For 2D input (batch_size, num_channels)
    
    auto output = torch::empty_like(input);
    
    dim3 block_size(256);
    dim3 grid_size(num_groups, batch_size);
    size_t shared_mem_size = 2 * sizeof(float);
    
    group_norm_forward_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        channels_per_group,
        elements_per_channel,
        eps
    );
    
    return output;
}
"""

group_norm_cpp_source = "torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps=1e-5);"

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