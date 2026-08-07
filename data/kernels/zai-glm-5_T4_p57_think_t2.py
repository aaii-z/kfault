import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU + HardSwish
fused_conv_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

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

    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate total number of output elements
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_elements) {
        // Decompose idx into (n, oc, oh, ow)
        int ow = idx % out_width;
        int tmp = idx / out_width;
        int oh = tmp % out_height;
        tmp = tmp / out_height;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;

        // Initialize sum with bias
        float sum = (bias != nullptr) ? bias[oc] : 0.0f;

        // Perform convolution
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int ih = oh + kh - padding;
                    const int iw = ow + kw - padding;

                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        const int input_idx = ((n * in_channels + ic) * height + ih) * width + iw;
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
        output[idx] = sum;
    }
}

torch::Tensor fused_conv_relu_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int padding) {

    // Get dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    // Calculate output dimensions (assuming stride=1, dilation=1)
    const int out_height = height + 2 * padding - kernel_size + 1;
    const int out_width = width + 2 * padding - kernel_size + 1;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Total number of threads needed
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    // Set up CUDA kernel launch parameters
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Get raw pointers
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    // Launch kernel
    fused_conv_relu_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
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
    "torch::Tensor fused_conv_relu_hardswish_cuda(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias, int padding);"
)

# Compile the inline CUDA code
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
        # We define a standard Conv2d to hold the weights and bias.
        # This ensures weight initialization matches the original model exactly.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_conv = fused_conv

    def forward(self, x):
        # Use the weights from the standard Conv2d layer
        # padding=0 is the default for nn.Conv2d
        return self.fused_conv.fused_conv_relu_hardswish_cuda(x, self.conv.weight, self.conv.bias, 0)