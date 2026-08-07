import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LeakyReLU, multiplication, and LeakyReLU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_ops_kernel(const float* __restrict__ x, const float* __restrict__ multiplier, float* __restrict__ out, int size, int channels, int channel_stride, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / channel_stride) % channels;
        float val = x[idx];
        // First LeakyReLU
        val = val > 0.0f ? val : val * negative_slope;
        // Multiplication
        val = val * multiplier[c];
        // Second LeakyReLU
        val = val > 0.0f ? val : val * negative_slope;
        out[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor multiplier, float negative_slope) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // Calculate channel stride (size per channel for one batch item)
    // x shape: [Batch, Channels, Depth, Height, Width]
    // channel_stride = Depth * Height * Width
    int channel_stride = x.numel() / (x.size(0) * x.size(1));
    int channels = x.size(1);

    fused_ops_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), multiplier.data_ptr<float>(), out.data_ptr<float>(), size, channels, channel_stride, negative_slope);

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor multiplier, float negative_slope);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, applies a fused operation
    (LeakyReLU, multiplication, LeakyReLU), and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused LeakyReLU + Multiplication + LeakyReLU
        x = self.fused_ops.fused_ops_cuda(x, self.multiplier, 0.2)
        x = self.max_pool(x)
        return x