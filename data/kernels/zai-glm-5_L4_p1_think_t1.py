import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv2d + ReLU + Bias
conv_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for Conv2d + ReLU + Bias
// Assumes stride=1, padding=0, dilation=1, groups=1
__global__ void conv_relu_bias_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ conv_bias, // Bias for conv
    const float* __restrict__ add_bias,  // Bias for addition
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size) {
    
    // Output dimensions
    int out_h = height - kernel_size + 1;
    int out_w = width - kernel_size + 1;
    
    // Calculate output pixel coordinates
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z % out_channels;
    int n = blockIdx.z / out_channels;

    if (w_out >= out_w || h_out >= out_h) return;

    // Initialize accumulator with conv bias
    float value = conv_bias[c_out];

    // Perform convolution
    // Input: (N, C_in, H, W), Weight: (C_out, C_in, kH, kW)
    // We iterate over input channels and kernel spatial dims
    
    // Pointers for the current batch and output channel
    const float* input_ptr = input + n * in_channels * height * width;
    const float* weight_ptr = weight + c_out * in_channels * kernel_size * kernel_size;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;
                
                float in_val = input_ptr[c_in * height * width + h_in * width + w_in];
                float w_val = weight_ptr[c_in * kernel_size * kernel_size + kh * kernel_size + kw];
                value += in_val * w_val;
            }
        }
    }

    // Apply ReLU
    value = fmaxf(0.0f, value);

    // Add the external bias
    // add_bias shape is (out_channels, 1, 1)
    value += add_bias[c_out];

    // Write output
    // Output shape: (N, C_out, out_h, out_w)
    output[n * out_channels * out_h * out_w + c_out * out_h * out_w + h_out * out_w + w_out] = value;
}

torch::Tensor conv_relu_bias_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor conv_bias, 
    torch::Tensor add_bias) {
    
    // Input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assuming square kernel

    // Output dimensions
    int out_h = height - kernel_size + 1;
    int out_w = width - kernel_size + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    // Block and Grid dimensions
    // Using 16x16 threads per block for spatial dimensions
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        batch_size * out_channels
    );

    conv_relu_bias_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        add_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width, kernel_size
    );

    return output;
}
"""

conv_relu_bias_cpp_source = (
    "torch::Tensor conv_relu_bias_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias, torch::Tensor add_bias);"
)

# Compile the inline CUDA code
conv_relu_bias_op = load_inline(
    name="conv_relu_bias",
    cpp_sources=conv_relu_bias_cpp_source,
    cuda_sources=conv_relu_bias_source,
    functions=["conv_relu_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for Conv2d + ReLU + Bias.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Initialize weights using nn.Conv2d to ensure correct initialization (kaiming uniform)
        # matching the reference model behavior.
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(conv.weight.data.clone())
        self.conv_bias = nn.Parameter(conv.bias.data.clone())
        
        # Additional bias
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Load the custom op
        self.conv_relu_bias = conv_relu_bias_op

    def forward(self, x):
        # Call the custom fused kernel
        return self.conv_relu_bias.conv_relu_bias_cuda(x, self.weight, self.conv_bias, self.bias)