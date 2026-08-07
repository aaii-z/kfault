import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU + HardSwish
fused_conv_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

// CUDA kernel for fused Conv2d + ReLU + HardSwish
__global__ void fused_conv_relu_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int padding) {

    // Calculate output pixel coordinates
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z % out_channels;
    const int b = blockIdx.z / out_channels;

    if (ow >= out_width || oh >= out_height || oc >= out_channels || b >= batch_size) return;

    // Initialize accumulator with bias
    float sum = bias ? bias[oc] : 0.0f;

    // Perform convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = oh + kh - padding;
                const int iw = ow + kw - padding;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    const int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                    const int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Apply ReLU
    sum = fmaxf(sum, 0.0f);

    // Apply HardSwish: x * clamp((x + 3) / 6, 0, 1)
    float hardswish_val = (sum + 3.0f) / 6.0f;
    hardswish_val = fmaxf(0.0f, fminf(1.0f, hardswish_val));
    sum = sum * hardswish_val;

    // Write output
    const int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
    output[output_idx] = sum;
}

torch::Tensor fused_conv_relu_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int padding) {

    // Get dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2); // Assuming square kernel

    // Calculate output dimensions
    const int out_height = height + 2 * padding - kernel_size + 1;
    const int out_width = width + 2 * padding - kernel_size + 1;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Set up CUDA kernel launch parameters
    const dim3 block_size(16, 16, 1);
    const dim3 grid_size(
        (out_width + block_size.x - 1) / block_size.x,
        (out_height + block_size.y - 1) / block_size.y,
        batch_size * out_channels
    );

    // Launch kernel
    fused_conv_relu_hardswish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        out_height,
        out_width,
        kernel_size,
        padding
    );

    return output;
}
"""

fused_conv_relu_hardswish_cpp_source = (
    "torch::Tensor fused_conv_relu_hardswish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int padding);"
)

# Compile the inline CUDA code for fused Conv2d + ReLU + HardSwish
fused_conv = load_inline(
    name="fused_conv_relu_hardswish",
    cpp_sources=fused_conv_relu_hardswish_cpp_source,
    cuda_sources=fused_conv_relu_hardswish_source,
    functions=["fused_conv_relu_hardswish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a fused convolution, ReLU, and HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Create weight and bias parameters similar to nn.Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize parameters similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_conv = fused_conv

    def forward(self, x):
        # Default nn.Conv2d padding is 0, which results in output size 126x126 for input 128x128 and kernel 3
        padding = 0
        return self.fused_conv.fused_conv_relu_hardswish_cuda(x, self.weight, self.bias, padding)