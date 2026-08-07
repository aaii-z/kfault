import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU + HardSwish
fused_conv_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

// Define block sizes
#define BLOCK_SIZE 16
#define KERNEL_SIZE 3
#define CHANNEL_BLOCK_SIZE 8

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
    int padding) {
    
    // Calculate output pixel coordinates
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tc = blockIdx.z * CHANNEL_BLOCK_SIZE + threadIdx.z;
    const int tn = blockIdx.w;
    
    if (tx >= out_width || ty >= out_height || tc >= out_channels) return;
    
    // Initialize accumulator for this output pixel
    float sum = 0.0f;
    
    // Shared memory for input tile
    __shared__ float shared_input[BLOCK_SIZE + KERNEL_SIZE - 1][BLOCK_SIZE + KERNEL_SIZE - 1];
    
    // Calculate input tile coordinates with padding
    const int in_x = tx - padding;
    const int in_y = ty - padding;
    
    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load input tile into shared memory
        for (int i = threadIdx.y; i < BLOCK_SIZE + KERNEL_SIZE - 1; i += blockDim.y) {
            for (int j = threadIdx.x; j < BLOCK_SIZE + KERNEL_SIZE - 1; j += blockDim.x) {
                int x = in_x + j;
                int y = in_y + i;
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    shared_input[i][j] = input[tn * in_channels * height * width + 
                                              ic * height * width + 
                                              y * width + x];
                } else {
                    shared_input[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();
        
        // Perform convolution for this input channel
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                float in_val = shared_input[threadIdx.y + ky][threadIdx.x + kx];
                float w_val = weight[tc * in_channels * KERNEL_SIZE * KERNEL_SIZE + 
                                    ic * KERNEL_SIZE * KERNEL_SIZE + 
                                    ky * KERNEL_SIZE + kx];
                sum += in_val * w_val;
            }
        }
        __syncthreads();
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
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid((out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
              out_channels);
    
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