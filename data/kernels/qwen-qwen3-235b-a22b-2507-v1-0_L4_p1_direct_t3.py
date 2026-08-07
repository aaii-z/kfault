import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + ReLU + BiasAdd
fused_conv_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_relu_bias_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int out_channels,
    int in_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    int weight_stride = in_channels * kernel_size * kernel_size;
    int input_stride = in_channels * input_height * input_width;
    int input_channel_stride = input_height * input_width;

    float sum = 0.0f;
    int weight_idx = c * weight_stride;

    for (int ci = 0; ci < in_channels; ++ci) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = h + kh - pad;
                int iw = w + kw - pad;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_offset = n * input_stride + ci * input_channel_stride + ih * input_width + iw;
                    sum += input[input_offset] * weight[weight_idx];
                }
                weight_idx++;
            }
        }
    }

    float bias_val = bias[c];
    float relu_val = fmaxf(0.0f, sum + bias_val);
    output[idx] = relu_val;
}

torch::Tensor fused_conv_relu_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int pad
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    // Correctly compute output dimensions for convolution with padding
    auto output_height = input_height + 2 * pad - kernel_size + 1;
    auto output_width = input_width + 2 * pad - kernel_size + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_relu_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        in_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        pad
    );

    return output;
}
"""

fused_conv_relu_bias_cpp_source = """
torch::Tensor fused_conv_relu_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int pad
);
"""

# Compile the inline CUDA kernel
fused_conv_relu_bias = load_inline(
    name="fused_conv_relu_bias",
    cpp_sources=fused_conv_relu_bias_cpp_source,
    cuda_sources=fused_conv_relu_bias_source,
    functions=["fused_conv_relu_bias_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused Conv2d + ReLU + BiasAdd in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pad = kernel_size // 2
        self.fused_op = fused_conv_relu_bias

    def forward(self, x):
        return self.fused_op.fused_conv_relu_bias_cuda(x, self.weight, self.bias, self.pad)