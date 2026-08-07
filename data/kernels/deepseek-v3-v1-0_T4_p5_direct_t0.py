import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv_transpose - bias - tanh
fused_conv_bias_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_conv_bias_tanh_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding) {
    
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_output_size = batch_size * out_channels * output_height * output_width;
    
    if (output_idx >= total_output_size) return;
    
    // Decompose output index
    const int n = output_idx / (out_channels * output_height * output_width);
    const int remainder = output_idx % (out_channels * output_height * output_width);
    const int c = remainder / (output_height * output_width);
    const int spatial_idx = remainder % (output_height * output_width);
    const int h_out = spatial_idx / output_width;
    const int w_out = spatial_idx % output_width;
    
    float sum = 0.0f;
    
    // Perform transposed convolution
    for (int kc = 0; kc < in_channels; ++kc) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int h_in = h_out + padding - kh * dilation;
                const int w_in = w_out + padding - kw * dilation;
                
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    const int input_idx = ((n * in_channels + kc) * input_height + h_in) * input_width + w_in;
                    const int weight_idx = ((kc * out_channels + c) * kernel_size + kh) * kernel_size + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Subtract bias and apply tanh
    const float biased = sum - bias[c];
    output[output_idx] = tanhf(biased);
}

torch::Tensor fused_conv_bias_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int output_padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(1);
    
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    
    const int total_output_size = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_output_size + block_size - 1) / block_size;
    
    fused_conv_bias_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding);
    
    return output;
}
"""

fused_conv_bias_tanh_cpp_source = "torch::Tensor fused_conv_bias_tanh_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int stride, int padding, int output_padding);"

# Compile the inline CUDA code
fused_conv_bias_tanh = load_inline(
    name="fused_conv_bias_tanh",
    cpp_sources=fused_conv_bias_tanh_cpp_source,
    cuda_sources=fused_conv_bias_tanh_source,
    functions=["fused_conv_bias_tanh_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store the custom operator
        self.fused_op = fused_conv_bias_tanh

    def forward(self, x):
        return self.fused_op.fused_conv_bias_tanh_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.output_padding
        )