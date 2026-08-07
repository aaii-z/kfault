import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_conv_div_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_div_relu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float divisor, float negative_slope,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int output_h, int output_w) {
    
    const int output_size = batch_size * out_channels * output_h * output_w;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    
    int n = idx / (out_channels * output_h * output_w);
    int remainder = idx % (out_channels * output_h * output_w);
    int c_out = remainder / (output_h * output_w);
    remainder = remainder % (output_h * output_w);
    int h_out = remainder / output_w;
    int w_out = remainder % output_w;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;
                
                int input_idx = n * (in_channels * height * width) + 
                              c_in * (height * width) + 
                              h_in * width + w_in;
                              
                int weight_idx = c_out * (in_channels * kernel_size * kernel_size) + 
                               c_in * (kernel_size * kernel_size) + 
                               kh * kernel_size + kw;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    sum = sum / divisor;
    
    if (sum < 0.0f) {
        sum = sum * negative_slope;
    }
    
    output[idx] = sum;
}

torch::Tensor fused_conv_div_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float divisor, float negative_slope) {
    
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
    
    const int output_h = height - kernel_size + 1;
    const int output_w = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * output_h * output_w;
    
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, 
                              input.options());
    
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    fused_conv_div_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        divisor,
        negative_slope,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        output_h,
        output_w);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_div_relu_cpp_source = """
torch::Tensor fused_conv_div_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float divisor, float negative_slope);
"""

fused_conv_div_relu = load_inline(
    name="fused_conv_div_relu",
    cpp_sources=fused_conv_div_relu_cpp_source,
    cuda_sources=fused_conv_div_relu_source,
    functions=["fused_conv_div_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.divisor = float(divisor)
        self.negative_slope = 0.01
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        nn.init.kaiming_uniform_(self.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.bias, 0.0)
        
        self.fused_op = fused_conv_div_relu

    def forward(self, x):
        return self.fused_op.fused_conv_div_relu_cuda(
            x, self.weight, self.bias, self.divisor, self.negative_slope
        )