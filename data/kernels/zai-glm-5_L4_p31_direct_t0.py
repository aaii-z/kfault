import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min, bias add, and scale
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void fused_ops_kernel(const float* __restrict__ x, float constant_value, const float* __restrict__ bias, float scaling_factor, float* __restrict__ out, int num_elements, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Calculate channel index to fetch the correct bias value
        // x is (N, C, H, W)
        int spatial_size = height * width;
        int c_idx = (idx / spatial_size) % channels;
        
        float val = x[idx];
        
        // Min operation
        val = std::min(val, constant_value);
        
        // Bias addition
        val += bias[c_idx];
        
        // Scaling
        val *= scaling_factor;
        
        out[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, float constant_value, torch::Tensor bias, float scaling_factor) {
    auto sizes = x.sizes();
    int num_elements = x.numel();
    auto out = torch::empty_like(x);
    
    int channels = sizes[1];
    int height = sizes[2];
    int width = sizes[3];

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        constant_value, 
        bias.data_ptr<float>(), 
        scaling_factor, 
        out.data_ptr<float>(), 
        num_elements,
        channels,
        height,
        width
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, float constant_value, torch::Tensor bias, float scaling_factor);"
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
    Optimized model that performs a convolution followed by a custom CUDA kernel
    fusing min, bias addition, and scaling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_ops_cuda(x, self.constant_value, self.bias, self.scaling_factor)
        return x