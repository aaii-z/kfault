import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel for ConvTranspose3d + Scale + BatchNorm + GlobalAvgPool
fused_conv_transpose_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>

#define CUDA_CHECK(call) {\\
    cudaError_t err = call;\\
    if (err != cudaSuccess) {\\
        fprintf(stderr, "CUDA error at %s:%d: %s\\\\n", __FILE__, __LINE__, cudaGetErrorString(err));\\
        exit(EXIT_FAILURE);\\
    }\\
}

__global__ void fused_conv_transpose_scale_bn_pool_kernel(
    const float* input, const float* weight, const float* bias, 
    const float* running_mean, const float* running_var, 
    float* intermediate_output, float scale_factor, float eps,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int output_depth, int output_height, int output_width) {
    
    const int output_size = output_depth * output_height * output_width;
    const int total_output_elements = batch_size * out_channels * output_size;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_output_elements) return;
    
    const int n = idx / (out_channels * output_size);
    const int c = (idx % (out_channels * output_size)) / output_size;
    const int spatial_idx = idx % output_size;
    const int d = spatial_idx / (output_height * output_width);
    const int h = (spatial_idx % (output_height * output_width)) / output_width;
    const int w = spatial_idx % output_width;
    
    const int padding = (kernel_depth - 1) / 2;
    float result = 0.0;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int input_d = d - kd + padding;
                    int input_h = h - kh + padding;
                    int input_w = w - kw + padding;
                    
                    if (input_d >= 0 && input_d < input_depth &&
                        input_h >= 0 && input_h < input_height &&
                        input_w >= 0 && input_w < input_width) {
                        
                        int input_idx = ((n * in_channels + ic) * input_depth + input_d) * input_height * input_width +
                                       input_h * input_width + input_w;
                        
                        int weight_idx = ((ic * out_channels + c) * kernel_depth + kd) * kernel_height * kernel_width +
                                        kh * kernel_width + kw;
                        
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    if (bias != nullptr) {
        result += bias[c];
    }
    
    result *= scale_factor;
    
    if (running_mean != nullptr && running_var != nullptr) {
        float mean = running_mean[c];
        float var = running_var[c];
        result = (result - mean) / sqrtf(var + eps);
    }
    
    intermediate_output[idx] = result;
}

__global__ void global_avg_pool_kernel(
    const float* input, float* output,
    int batch_size, int out_channels,
    int output_depth, int output_height, int output_width) {
    
    const int spatial_size = output_depth * output_height * output_width;
    const int total_elements = batch_size * out_channels;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    const int n = idx / out_channels;
    const int c = idx % out_channels;
    
    float sum = 0.0;
    const float* channel_data = input + (n * out_channels + c) * spatial_size;
    
    for (int i = 0; i < spatial_size; ++i) {
        sum += channel_data[i];
    }
    
    output[n * out_channels + c] = sum / static_cast<float>(spatial_size);
}

torch::Tensor fused_conv_transpose_scale_bn_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    float scale_factor, float eps) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);
    
    int output_depth = input_depth;
    int output_height = input_height;
    int output_width = input_width;
    
    const int output_size = output_depth * output_height * output_width;
    const int total_output_elements = batch_size * out_channels * output_size;
    
    auto intermediate_output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                                           input.options());
    
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;
    
    fused_conv_transpose_scale_bn_pool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        running_mean.defined() ? running_mean.data_ptr<float>() : nullptr,
        running_var.defined() ? running_var.data_ptr<float>() : nullptr,
        intermediate_output.data_ptr<float>(),
        scale_factor, eps,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        output_depth, output_height, output_width);
    
    CUDA_CHECK(cudaGetLastError());
    
    auto final_output = torch::empty({batch_size, out_channels, 1, 1, 1}, input.options());
    const int pool_elements = batch_size * out_channels;
    const int pool_blocks = (pool_elements + block_size - 1) / block_size;
    
    global_avg_pool_kernel<<<pool_blocks, block_size>>>(
        intermediate_output.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size, out_channels,
        output_depth, output_height, output_width);
    
    CUDA_CHECK(cudaGetLastError());
    
    return final_output;
}
"""

fused_conv_transpose_ops_cpp_source = """
torch::Tensor fused_conv_transpose_scale_bn_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    float scale_factor, float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_transpose_ops",
    cpp_sources=fused_conv_transpose_ops_cpp_source,
    cuda_sources=fused_conv_transpose_ops_source,
    functions=["fused_conv_transpose_scale_bn_pool_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.scale_factor = scale_factor
        self.eps = eps
        
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        self.running_mean = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        self.running_var = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        self.num_batches_tracked = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)
        
        self.reset_parameters()
        self.fused_ops = fused_ops

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)
        self.num_batches_tracked.zero_()

    def forward(self, x):
        return self.fused_ops.fused_conv_transpose_scale_bn_pool_cuda(
            x, self.weight, self.bias, 
            self.running_mean, self.running_var,
            self.scale_factor, self.eps
        )