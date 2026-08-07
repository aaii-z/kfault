import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + Division + LeakyReLU
fused_conv_div_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

// Define constants for LeakyReLU
#define NEGATIVE_SLOPE 0.01f

// Kernel for fused Division and LeakyReLU
__global__ void div_leaky_relu_kernel(const float* __restrict__ input, float* __restrict__ output, int size, float inv_divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * inv_divisor;
        output[idx] = val > 0.0f ? val : val * NEGATIVE_SLOPE;
    }
}

// Helper function to launch the elementwise kernel
void launch_div_leaky_relu(const float* input, float* output, int size, float inv_divisor, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    div_leaky_relu_kernel<<<num_blocks, block_size, 0, stream>>>(input, output, size, inv_divisor);
}

// CUDA wrapper function
torch::Tensor fused_conv_div_leaky_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    float divisor) {

    // 1. Perform Conv2d using cuDNN (highly optimized)
    // We use conv2d which handles padding and stride internally
    // Note: F::conv2d in C++ dispatches to cuDNN
    auto conv_options = torch::nn::functional::Conv2dFuncOptions()
                            .stride(stride)
                            .padding(padding);
    
    // We need to handle the optional bias
    torch::Tensor conv_out;
    if (bias.defined() && bias.numel() > 0) {
         conv_out = torch::conv2d(x, weight, bias, conv_options);
    } else {
         conv_out = torch::conv2d(x, weight, torch::Tensor(), conv_options);
    }

    // 2. Fused Division and LeakyReLU
    // Create output tensor (we can operate in-place if we don't need the intermediate conv_out, 
    // but PyTorch manages memory. Let's create a new tensor for the result).
    auto out = torch::empty_like(conv_out);
    int size = conv_out.numel();
    float inv_divisor = 1.0f / divisor;

    // Get the current CUDA stream to ensure synchronization
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    launch_div_leaky_relu(conv_out.data_ptr<float>(), out.data_ptr<float>(), size, inv_divisor, stream);

    return out;
}
"""

fused_conv_div_leaky_relu_cpp_source = """
torch::Tensor fused_conv_div_leaky_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    float divisor);
"""

# Compile the inline CUDA code
fused_conv_op = load_inline(
    name="fused_conv_div_leaky_relu",
    cpp_sources=fused_conv_div_leaky_relu_cpp_source,
    cuda_sources=fused_conv_div_leaky_relu_source,
    functions=["fused_conv_div_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for Conv2d + Division + LeakyReLU fusion.
    The Conv2d uses PyTorch's internal cuDNN implementation, while the subsequent
    element-wise operations are fused into a single custom kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        # We keep the Conv2d module to hold weights and bias easily
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.fused_op = fused_conv_op

    def forward(self, x):
        # Extract weights and bias from the Conv2d module
        weight = self.conv.weight
        bias = self.conv.bias
        
        # Call the custom fused CUDA operator
        # We pass stride and padding from the conv module
        return self.fused_op.fused_conv_div_leaky_relu_cuda(
            x, 
            weight, 
            bias, 
            self.conv.stride[0], 
            self.conv.padding[0], 
            self.divisor
        )