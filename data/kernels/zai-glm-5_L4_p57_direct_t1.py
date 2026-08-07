import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU + HardSwish
fused_conv_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel for fused Conv2d + ReLU + HardSwish
__global__ void conv_relu_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int out_height,
    int out_width,
    int kernel_size,
    int padding) {
    
    // Calculate output pixel coordinates
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tc = blockIdx.z;
    const int tn = blockIdx.z / out_channels; // This logic is flawed for batch, fixing below
    
    // Correct indexing for batch and channel
    // We map grid.z to (batch_index, out_channel_index)
    // Let's use a simple 1D index for z to cover batch * out_channels
    const int batch_channel_idx = blockIdx.z;
    const int tn = batch_channel_idx / out_channels;
    const int tc = batch_channel_idx % out_channels;
    
    if (tx >= out_width || ty >= out_height || tn >= batch_size) return;
    
    // Initialize accumulator for this output pixel
    float sum = 0.0f;
    
    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Perform convolution for this input channel
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Calculate input coordinates
                int in_x = tx + kx - padding;
                int in_y = ty + ky - padding;
                
                // Check bounds
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    float in_val = input[tn * in_channels * height * width + 
                                         ic * height * width + 
                                         in_y * width + in_x];
                    
                    float w_val = weight[tc * in_channels * kernel_size * kernel_size + 
                                        ic * kernel_size * kernel_size + 
                                        ky * kernel_size + kx];
                    sum += in_val * w_val;
                }
            }
        }
    }
    
    // Add bias if present
    if (bias != nullptr) {
        sum += bias[tc];
    }
    
    // Apply ReLU
    sum = fmaxf(sum, 0.0f);
    
    // Apply HardSwish: x * clamp((x + 3) / 6, 0, 1)
    float hardswish_val = sum * fminf(fmaxf((sum + 3.0f) / 6.0f, 0.0f), 1.0f);
    
    // Write output
    output[tn * out_channels * out_height * out_width + 
           tc * out_height * out_width + 
           ty * out_width + tx] = hardswish_val;
}

torch::Tensor conv_relu_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int padding) {
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Calculate output dimensions
    int out_height = height + 2 * padding - kernel_size + 1;
    int out_width = width + 2 * padding - kernel_size + 1;
    
    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    // Set up grid and block dimensions
    dim3 block(16, 16, 1);
    dim3 grid((out_width + 15) / 16,
              (out_height + 15) / 16,
              batch_size * out_channels);
    
    // Launch kernel
    conv_relu_hardswish_kernel<<<grid, block>>>(
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
        padding);
    
    return output;
}
"""

fused_conv_relu_hardswish_cpp_source = """
torch::Tensor conv_relu_hardswish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int padding);
"""

# Compile the inline CUDA code for fused Conv2d + ReLU + HardSwish
fused_conv_relu_hardswish = load_inline(
    name="fused_conv_relu_hardswish",
    cpp_sources=fused_conv_relu_hardswish_cpp_source,
    cuda_sources=fused_conv_relu_hardswish_source,
    functions=["conv_relu_hardswish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies ReLU, and applies HardSwish activation
    using a custom fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize the custom CUDA kernel
        self.fused_conv = fused_conv_relu_hardswish

    def forward(self, x):
        return self.fused_conv.conv_relu_hardswish_cuda(x, self.weight, self.bias, self.padding)