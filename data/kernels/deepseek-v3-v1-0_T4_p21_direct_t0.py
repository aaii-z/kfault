import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv + bias + scale + sigmoid + groupnorm
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_conv_bias_scale_sigmoid_groupnorm_kernel(
    const float* input, const float* weight, const float* bias, const float* scale,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int num_groups,
    int input_stride, int output_stride, int weight_stride) {
    
    const int group_size = out_channels / num_groups;
    const int spatial_size = height * width;
    const int total_elements = batch_size * out_channels * spatial_size;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Calculate indices
    int b = idx / (out_channels * spatial_size);
    int c = (idx % (out_channels * spatial_size)) / spatial_size;
    int hw = idx % spatial_size;
    int h = hw / width;
    int w = hw % width;
    
    int group_id = c / group_size;
    int group_start = group_id * group_size;
    int group_end = (group_id + 1) * group_size;
    
    // Convolution
    float conv_result = 0.0f;
    int k_half = kernel_size / 2;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h + kh - k_half;
                int w_in = w + kw - k_half;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = b * input_stride + ic * height * width + h_in * width + w_in;
                    int weight_idx = c * weight_stride + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                    conv_result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias and scale
    float result = (conv_result + bias[c]) * scale[c];
    
    // Sigmoid
    result = 1.0f / (1.0f + expf(-result));
    
    // Group normalization statistics (compute mean and variance for the group)
    __shared__ float group_sum[256];
    __shared__ float group_sq_sum[256];
    
    if (threadIdx.x < 256) {
        group_sum[threadIdx.x] = 0.0f;
        group_sq_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    atomicAdd(&group_sum[group_id], result);
    atomicAdd(&group_sq_sum[group_id], result * result);
    __syncthreads();
    
    float group_mean = group_sum[group_id] / (batch_size * group_size * spatial_size);
    float group_variance = group_sq_sum[group_id] / (batch_size * group_size * spatial_size) - group_mean * group_mean;
    group_variance = max(group_variance, 1e-5f);
    
    // Apply group normalization
    result = (result - group_mean) / sqrtf(group_variance);
    
    output[idx] = result;
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor scale, int num_groups) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = sqrt(weight.size(1) / in_channels);
    
    auto output = torch::empty({batch_size, out_channels, height, width}, input.options());
    
    int input_stride = in_channels * height * width;
    int output_stride = out_channels * height * width;
    int weight_stride = in_channels * kernel_size * kernel_size;
    
    const int total_elements = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_bias_scale_sigmoid_groupnorm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        scale.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width, kernel_size, num_groups,
        input_stride, output_stride, weight_stride);
    
    return output;
}
"""

fused_operator_cpp_source = "torch::Tensor fused_operator_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor scale, int num_groups);"

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels * kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.num_groups = num_groups
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(x, self.weight, self.bias, self.scale, self.num_groups)