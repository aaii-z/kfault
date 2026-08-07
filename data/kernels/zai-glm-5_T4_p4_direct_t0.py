import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + Mish + Mish
# We fuse the convolution with the two subsequent Mish activations.
# Mish is defined as: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
# We use the standard library function for the activation part.

fused_conv_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Define a helper for Mish activation: x * tanh(ln(1 + exp(x)))
__device__ __forceinline__ float mish_activation(float x) {
    // softplus(x) = ln(1 + exp(x))
    // We use logf and expf for precision.
    // Adding a check to avoid log(0) if expf(x) underflows, though softplus handles it naturally.
    // For large negative x, softplus(x) -> 0, tanh(0) -> 0, mish -> 0.
    // For large positive x, softplus(x) -> x, tanh(x) -> 1, mish -> x.
    return x * tanhf(log1pf(expf(x)));
}

// Kernel for fused Conv2d + Mish + Mish
// We use implicit GEMM approach where each thread computes one output pixel.
// This allows us to apply the activation function on the fly without storing the intermediate convolution result.
__global__ void conv2d_mish_mish_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels, 
    int height, int width, int kernel_size, 
    int stride, int padding, int dilation) {
    
    // Calculate output dimensions
    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_h * out_w;

    if (idx < total_elements) {
        // Decompose idx into (n, oc, oh, ow)
        int ow = idx % out_w;
        int oh = (idx / out_w) % out_h;
        int oc = (idx / (out_w * out_h)) % out_channels;
        int n = idx / (out_w * out_h * out_channels);

        float sum = 0.0f;

        // Apply convolution
        // Loop over input channels and kernel spatial dimensions
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Calculate input coordinates
                    int ih = oh * stride - padding + kh * dilation;
                    int iw = ow * stride - padding + kw * dilation;

                    // Check boundaries (padding)
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        int input_idx = n * (in_channels * height * width) + 
                                        ic * (height * width) + 
                                        ih * width + 
                                        iw;
                        
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size) + 
                                         ic * (kernel_size * kernel_size) + 
                                         kh * kernel_size + 
                                         kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias if present
        if (bias != nullptr) {
            sum += bias[oc];
        }

        // Apply Mish twice
        sum = mish_activation(mish_activation(sum));

        // Write output
        output[idx] = sum;
    }
}

// Helper function to launch the kernel
torch::Tensor fused_conv_mish_mish_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int stride, int padding, int dilation, int groups) {
    
    // Input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    // Output dimensions
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assuming square kernel
    
    // Allocate output tensor
    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    
    // Launch kernel
    int total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // We assume groups=1 for this specific optimization based on the problem description
    conv2d_mish_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width, kernel_size,
        stride, padding, dilation
    );
    
    return output;
}
"""

fused_conv_mish_mish_cpp_source = (
    "torch::Tensor fused_conv_mish_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation, int groups);"
)

# Compile the inline CUDA code
fused_conv_mish_mish_op = load_inline(
    name="fused_conv_mish_mish",
    cpp_sources=fused_conv_mish_mish_cpp_source,
    cuda_sources=fused_conv_mish_source,
    functions=["fused_conv_mish_mish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution followed by two Mish activations
    using a custom fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # We keep the Conv2d layer definition to hold the weights and bias
        # but we will not use its forward pass directly.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_op = fused_conv_mish_mish_op

    def forward(self, x):
        # Use the custom fused kernel
        # We extract the parameters from the defined Conv2d layer
        return self.fused_op.fused_conv_mish_mish_cuda(
            x, 
            self.conv.weight, 
            self.conv.bias, 
            self.conv.stride[0], 
            self.conv.padding[0], 
            self.conv.dilation[0], 
            self.conv.groups
        )