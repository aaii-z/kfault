import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose3d + Clamp + Division
fused_conv_clamp_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_transpose_clamp_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor,
    const int output_depth,
    const int output_height,
    const int output_width
) {
    const int output_size = batch_size * out_channels * output_depth * output_height * output_width;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= output_size) return;

    // Calculate output indices
    const int n = index / (out_channels * output_depth * output_height * output_width);
    const int c = (index / (output_depth * output_height * output_width)) % out_channels;
    const int d = (index / (output_height * output_width)) % output_depth;
    const int h = (index / output_width) % output_height;
    const int w = index % output_width;

    float value = bias ? bias[c] : 0.0f;

    // Iterate over kernel and input channels
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int input_d = d + padding - kd;
                const int input_h = h + padding - kh;
                const int input_w = w + padding - kw;
                
                if (input_d % stride == 0 && input_h % stride == 0 && input_w % stride == 0) {
                    const int strided_d = input_d / stride;
                    const int strided_h = input_h / stride;
                    const int strided_w = input_w / stride;
                    
                    if (strided_d >= 0 && strided_d < input_depth &&
                        strided_h >= 0 && strided_h < input_height &&
                        strided_w >= 0 && strided_w < input_width) {
                        for (int ic = 0; ic < in_channels; ic++) {
                            const int input_index = ((n * in_channels + ic) * input_depth + strided_d) * input_height * input_width +
                                                  strided_h * input_width + strided_w;
                            const int weight_index = ((ic * out_channels + c) * kernel_size + kd) * kernel_size * kernel_size +
                                                   kh * kernel_size + kw;
                            value += input[input_index] * weight[weight_index];
                        }
                    }
                }
            }
        }
    }

    // Apply clamp and division
    value = fmaxf(value, min_value);
    value /= divisor;
    
    output[index] = value;
}

torch::Tensor fused_conv_transpose_clamp_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    const int out_channels = weight.size(1);

    // Calculate output dimensions
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;

    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int output_size = batch_size * out_channels * output_depth * output_height * output_width;
    const int threads = 256;
    const int blocks = (output_size + threads - 1) / threads;

    fused_conv_transpose_clamp_div_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        min_value,
        divisor,
        output_depth,
        output_height,
        output_width
    );

    return output;
}
"""

fused_conv_clamp_div_cpp_source = """
torch::Tensor fused_conv_transpose_clamp_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor
);
"""

# Compile the inline CUDA code
fused_conv_clamp_div = load_inline(
    name="fused_conv_clamp_div",
    cpp_sources=fused_conv_clamp_div_cpp_source,
    cuda_sources=fused_conv_clamp_div_source,
    functions=["fused_conv_transpose_clamp_div_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = min_value
        self.divisor = divisor
        
        # Initialize weight and bias similar to nn.ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # Reset parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size * self.kernel_size
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_conv_clamp_div.fused_conv_transpose_clamp_div_cuda(
            x, self.weight, self.bias, self.kernel_size, self.stride, self.padding, self.min_value, self.divisor
        )