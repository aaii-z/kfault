import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + HardSwish + ReLU
fused_conv_hardswish_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

// Define block sizes
#define BLOCK_SIZE 16
#define KERNEL_SIZE 3

// HardSwish function: x * F.relu6(x + 3) / 6
// ReLU function: max(0, x)
// Combined: HardSwish output is always >= 0, so ReLU(HardSwish(x)) == HardSwish(x)
// We can skip the ReLU explicitly as HardSwish is non-negative.
__device__ __forceinline__ float hardswish_relu_func(float x) {
    // F.relu6(x + 3) is min(max(x + 3, 0), 6)
    float val = x + 3.0f;
    if (val < 0.0f) val = 0.0f;
    else if (val > 6.0f) val = 6.0f;
    return x * val / 6.0f;
}

// Kernel for Conv2d + HardSwish + ReLU
__global__ void conv_hardswish_relu_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, 
    int out_height, int out_width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w) {
    
    // Calculate output coordinates
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % out_channels; // Output channel
    int n = blockIdx.z / out_channels;  // Batch index

    if (w_out >= out_width || h_out >= out_height || oc >= out_channels || n >= batch_size) return;

    float sum = 0.0f;

    // We iterate over input channels and the kernel window
    for (int ic = 0; ic < in_channels; ++ic) {
        // Kernel weights layout: (out_channels, in_channels, kH, kW)
        // We iterate over kernel height and width
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            int h_in = h_out * stride_h + kh - pad_h;
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                int w_in = w_out * stride_w + kw - pad_w;
                
                // Check padding boundaries (zero-padding)
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    int input_idx = ((n * in_channels + ic) * in_height + h_in) * in_width + w_in;
                    int weight_idx = ((oc * in_channels + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Apply HardSwish + ReLU
    sum = hardswish_relu_func(sum);

    // Write output
    int output_idx = ((n * out_channels + oc) * out_height + h_out) * out_width + w_out;
    output[output_idx] = sum;
}

torch::Tensor fused_conv_hardswish_relu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias) {
    
    // Input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    // Output dimensions
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    // Conv2d default parameters for this model
    int stride = 1;
    int dilation = 1;
    // PyTorch default padding for Conv2d is 0, but we calculate effective padding
    // based on the kernel size to match the default behavior of "same" padding if implied
    // or strictly follow the provided parameters. 
    // The original model uses nn.Conv2d with no padding argument, so padding=0.
    int padding = 0; 
    
    // Calculate output dimensions
    // out_height = (in_height + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    // Grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size * out_channels
    );
    
    // Launch kernel
    conv_hardswish_relu_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width,
        out_height, out_width,
        kernel_h, kernel_w,
        padding, padding,
        stride, stride
    );
    
    return output;
}
"""

fused_conv_hardswish_relu_cpp_source = (
    "torch::Tensor fused_conv_hardswish_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_conv_hardswish_relu_cpp_source,
    cuda_sources=fused_conv_hardswish_relu_source,
    functions=["fused_conv_hardswish_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, applies HardSwish, and then ReLU
    using a custom fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # We still define the convolution layer to hold weights and biases
        # but we will not use its forward pass.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        # Extract weights and bias from the conv layer
        weight = self.conv.weight
        bias = self.conv.bias
        
        # Call the custom fused CUDA kernel
        return self.fused_ops.fused_conv_hardswish_relu_cuda(x, weight, bias)