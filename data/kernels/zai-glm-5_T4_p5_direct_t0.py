import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias subtraction and tanh activation
fused_bias_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bias_tanh_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out, int size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Calculate indices
        int n = idx / (channels * height * width);
        int c = (idx / (height * width)) % channels;
        
        // Subtract bias and apply tanh
        float val = x[idx] - bias[c];
        out[idx] = tanhf(val);
    }
}

torch::Tensor fused_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // Get dimensions for bias indexing
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    fused_bias_tanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size, 
        channels, 
        height, 
        width
    );

    return out;
}
"""

fused_bias_tanh_cpp_source = (
    "torch::Tensor fused_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused bias subtraction and tanh
fused_bias_tanh_op = load_inline(
    name="fused_bias_tanh",
    cpp_sources=fused_bias_tanh_cpp_source,
    cuda_sources=fused_bias_tanh_source,
    functions=["fused_bias_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    The bias subtraction and tanh activation are fused into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.fused_bias_tanh = fused_bias_tanh_op

    def forward(self, x):
        x = self.conv_transpose(x)
        # Use the custom fused kernel for bias subtraction and tanh
        x = self.fused_bias_tanh.fused_bias_tanh_cuda(x, self.bias)
        return x