import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + GELU + AdaptiveAvgPool2d
fused_conv_gelu_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Define constants for GELU
const float GELU_CONST = 0.7978845608028654f; // sqrt(2/pi)
const float GELU_CONST2 = 0.044715f;

// Kernel for fused Conv2d + GELU + Global Average Pooling
// We assume the output spatial size is 1x1 (Global Average Pooling)
// This allows us to write directly to the final output buffer without intermediate large tensors.
__global__ void fused_conv_gelu_pool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int padding,
    int stride) {

    // Calculate output dimensions
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    float pool_scale = 1.0f / (out_h * out_w);

    // Calculate global thread index
    // Each thread computes one output element (batch, out_channel)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels;

    if (idx < total_elements) {
        int b = idx / out_channels;
        int oc = idx % out_channels;

        float sum = 0.0f;

        // Iterate over the spatial dimensions of the convolution output
        // This corresponds to the pooling window
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float conv_val = 0.0f;

                // Perform convolution for this specific output spatial location
                // We inline the convolution logic here
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int input_idx = b * (in_channels * height * width) + ic * (height * width) + ih * width + iw;
                                int weight_idx = oc * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + kh * kernel_size + kw;
                                conv_val += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }

                // Add bias
                conv_val += bias[oc];

                // Apply GELU activation
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                float gelu_val = 0.5f * conv_val * (1.0f + tanhf(GELU_CONST * (conv_val + GELU_CONST2 * conv_val * conv_val * conv_val)));

                // Accumulate for average pooling
                sum += gelu_val;
            }
        }

        // Write final result
        output[idx] = sum * pool_scale;
    }
}

torch::Tensor fused_conv_gelu_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int padding,
    int stride) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels + block_size - 1) / block_size;

    fused_conv_gelu_pool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        padding,
        stride
    );

    return output;
}
"""

fused_conv_gelu_pool_cpp_source = (
    "torch::Tensor fused_conv_gelu_pool_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int padding, int stride);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_conv_gelu_pool",
    cpp_sources=fused_conv_gelu_pool_cpp_source,
    cuda_sources=fused_conv_gelu_pool_source,
    functions=["fused_conv_gelu_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies GELU, and then performs global average pooling
    using a single fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # We still define the Conv2d layer to hold weights and bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        # Extract weights and bias from the Conv2d layer
        weight = self.conv.weight
        bias = self.conv.bias
        
        # Call the fused CUDA kernel
        # We use the padding and stride from the Conv2d layer
        return self.fused_op.fused_conv_gelu_pool_cuda(x, weight, bias, self.conv.padding[0], self.conv.stride[0])