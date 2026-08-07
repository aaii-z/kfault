import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution + ReLU + HardSwish
fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_activation_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride, int padding) {
    
    const int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    const int n = blockIdx.x;
    const int oc = blockIdx.y;
    const int oh = blockIdx.z / output_w;
    const int ow = blockIdx.z % output_w;
    
    if (n >= batch_size || oc >= out_channels || oh >= output_h || ow >= output_w) {
        return;
    }
    
    float sum = 0.0f;
    if (bias) {
        sum = bias[oc];
    }
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int ih = oh * stride + kh - padding;
                const int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    const int input_idx = ((n * in_channels + ic) * height + ih) * width + iw;
                    const int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // ReLU + HardSwish fusion
    const float relu_val = fmaxf(0.0f, sum);
    const float hardswish_val = relu_val * fminf(fmaxf((relu_val + 3.0f) / 6.0f, 0.0f), 1.0f);
    
    const int output_idx = ((n * out_channels + oc) * output_h + oh) * output_w + ow;
    output[output_idx] = hardswish_val;
}

torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) {
        CHECK_INPUT(bias);
    }
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, 
                              input.options().device(torch::kCUDA));
    
    const int num_blocks_x = batch_size;
    const int num_blocks_y = out_channels;
    const int num_blocks_z = output_h * output_w;
    
    dim3 grid_size(num_blocks_x, num_blocks_y, num_blocks_z);
    const int threads_per_block = 1;
    
    fused_conv_activation_kernel<<<grid_size, threads_per_block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, stride, padding);
    
    return output;
}
"""

fused_conv_activation_cpp_source = """
torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding);
"""

# Compile the inline CUDA code
fused_conv_activation = load_inline(
    name="fused_conv_activation",
    cpp_sources=fused_conv_activation_cpp_source,
    cuda_sources=fused_conv_activation_source,
    functions=["fused_conv_activation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = 0  # No padding to match original output size
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights using Kaiming initialization for ReLU
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        
        self.fused_op = fused_conv_activation

    def forward(self, x):
        return self.fused_op.fused_conv_activation_cuda(
            x, self.weight, self.bias, self.stride, self.padding
        )