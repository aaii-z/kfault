import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d, subtraction, and Mish activation
fused_conv_subtract_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Mish activation: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
__device__ float mish_activation(float x) {
    float sp = log1pf(expf(x)); // softplus: ln(1 + exp(x))
    return x * tanhf(sp);
}

__global__ void fused_conv_subtract_mish_kernel(
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
    float subtract_value_1,
    float subtract_value_2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    int input_h = input_height;
    int input_w = input_width;
    int pad = kernel_size / 2;
    int kernel_radius = kernel_size / 2;

    // Convolution computation
    float conv_val = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = h + kh - kernel_radius;
                int iw = w + kw - kernel_radius;

                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                    int input_idx = n * in_channels * input_h * input_w +
                                    ci * input_h * input_w +
                                    ih * input_w + iw;
                    int weight_idx = c * in_channels * kernel_size * kernel_size +
                                     ci * kernel_size * kernel_size +
                                     kh * kernel_size + kw;
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    conv_val += bias[c];

    // Subtract two values
    float subtracted = conv_val - subtract_value_1 - subtract_value_2;

    // Apply Mish activation
    output[idx] = mish_activation(subtracted);
}
"""

fused_conv_subtract_mish_cpp_source = """
torch::Tensor fused_conv_subtract_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value_1,
    float subtract_value_2,
    int output_height,
    int output_width);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_subtract_mish",
    cpp_sources=fused_conv_subtract_mish_cpp_source,
    cuda_sources=fused_conv_subtract_mish_source,
    functions=["fused_conv_subtract_mish_cuda"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Conv2d, subtraction, and Mish activation in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape
        output_height = input_height  # assuming padding is applied to preserve size
        output_width = input_width

        return fused_op.fused_conv_subtract_mish_cuda(
            x,
            self.weight,
            self.bias,
            self.subtract_value_1,
            self.subtract_value_2,
            output_height,
            output_width
        )