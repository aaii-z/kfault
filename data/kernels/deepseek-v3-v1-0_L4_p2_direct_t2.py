import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operation
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose_bias_clamp_scale_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float scaling_factor,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride, int padding, int output_padding) {
    
    const int output_size = batch_size * out_channels * output_height * output_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= output_size) return;
    
    // Decompose index
    const int n = idx / (out_channels * output_height * output_width);
    const int c = (idx / (output_height * output_width)) % out_channels;
    const int h = (idx / output_width) % output_height;
    const int w = idx % output_width;
    
    float sum = 0.0f;
    
    // Calculate corresponding input position for transposed convolution
    const int input_h_start = h - padding;
    const int input_w_start = w - padding;
    
    for (int kc = 0; kc < in_channels; ++kc) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int input_h = input_h_start + kh;
                const int input_w = input_w_start + kw;
                
                if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                    const int input_idx = ((n * in_channels + kc) * input_height + input_h) * input_width + input_w;
                    const int weight_idx = ((c * in_channels + kc) * kernel_size + kh) * kernel_size + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c];
    
    // Clamp, scale, clamp, and scale back in one go
    float result = fminf(1.0f, fmaxf(0.0f, sum));
    result = result * scaling_factor;
    result = fminf(1.0f, fmaxf(0.0f, result));
    result = result / scaling_factor;
    
    output[idx] = result;
}

torch::Tensor fused_conv_transpose_bias_clamp_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, int kernel_size, int stride, 
    int padding, int output_padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(0);
    
    // Calculate output dimensions
    const int output_height = (input_height - 1) * stride + kernel_size - 2 * padding + output_padding;
    const int output_width = (input_width - 1) * stride + kernel_size - 2 * padding + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
                              input.options());
    
    const int output_size = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    fused_conv_transpose_bias_clamp_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), scaling_factor,
        batch_size, in_channels, out_channels,
        input_height, input_width, output_height, output_width,
        kernel_size, stride, padding, output_padding);
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_conv_transpose_bias_clamp_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, int kernel_size, int stride, 
    int padding, int output_padding);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_transpose_bias_clamp_scale_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_conv_transpose_bias_clamp_scale_cuda(
            x, self.weight, self.bias, self.scaling_factor, 
            self.kernel_size, self.stride, self.padding, self.output_padding
        )