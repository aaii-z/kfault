import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias subtraction and tanh activation
fused_bias_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bias_tanh_kernel(const float* __restrict__ input, const float* __restrict__ bias, float* __restrict__ output, int N, int C, int H, int W) {
    // Calculate global thread index
    // We use a 3D grid: blockIdx.z for batch, blockIdx.y for channels, blockIdx.x for spatial dimensions
    int n = blockIdx.z;
    int c = blockIdx.y;
    
    int spatial_size = H * W;
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (spatial_idx < spatial_size) {
        int index = (n * C + c) * spatial_size + spatial_idx;
        float val = input[index];
        float b = bias[c]; // bias has shape (C, 1, 1), so we index with channel c
        output[index] = tanhf(val - b);
    }
}

torch::Tensor fused_bias_tanh_cuda(torch::Tensor input, torch::Tensor bias) {
    // input shape: (N, C, H, W)
    // bias shape: (C, 1, 1)
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int block_size = 1024;
    int spatial_size = H * W;
    int grid_x = (spatial_size + block_size - 1) / block_size;
    
    // Grid dimensions: (spatial_blocks, C, N)
    dim3 grid(grid_x, C, N);
    dim3 block(block_size);
    
    fused_bias_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N, C, H, W
    );
    
    return output;
}
"""

fused_bias_tanh_cpp_source = (
    "torch::Tensor fused_bias_tanh_cuda(torch::Tensor input, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused bias and tanh
fused_bias_tanh = load_inline(
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
    The bias subtraction and tanh activation are fused into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.fused_bias_tanh = fused_bias_tanh

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_bias_tanh.fused_bias_tanh_cuda(x, self.bias)
        return x