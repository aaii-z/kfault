import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias, scale, and sigmoid
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bias_scale_sigmoid_kernel(const float* __restrict__ x, const float* __restrict__ bias, const float* __restrict__ scale, float* __restrict__ out, int size, int channels, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int hw = h * w;
        int c_idx = (idx / hw) % channels;
        
        float val = x[idx];
        val += bias[c_idx];
        val *= scale[c_idx];
        // Sigmoid function: 1 / (1 + exp(-x))
        out[idx] = 1.0f / (1.0f + expf(-val));
    }
}

torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor x, torch::Tensor bias, torch::Tensor scale) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    int channels = x.size(1);
    int h = x.size(2);
    int w = x.size(3);
    
    fused_bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), out.data_ptr<float>(), size, channels, h, w
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor x, torch::Tensor bias, torch::Tensor scale);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_bias_scale_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, followed by a custom CUDA kernel
    for fused bias, scale, and sigmoid, and finally group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_bias_scale_sigmoid_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x