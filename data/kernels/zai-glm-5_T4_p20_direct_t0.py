import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_ops_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out, int size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / spatial_size) % channels;
        float original_x = x[idx];
        float b = bias[c];
        float x_plus_b = original_x + b;
        float x_plus_original = x_plus_b + original_x;
        float x_mul_original = x_plus_original * original_x;
        out[idx] = x_mul_original + original_x;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    int channels = x.size(1);
    int spatial_size = x.size(2) * x.size(3) * x.size(4);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), size, channels, spatial_size);

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused operations
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
    Optimized model that performs a 3D transposed convolution, followed by a fused kernel
    that combines addition, residual addition, multiplication, and another residual addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias)
        return x