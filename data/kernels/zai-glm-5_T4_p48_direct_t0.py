import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: scaling, tanh, bias multiplication, and sigmoid
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_ops_kernel(const float* __restrict__ x, const float* __restrict__ scaling_factor, const float* __restrict__ bias, float* __restrict__ out, int size, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Calculate the channel index to access the correct scaling factor and bias
        // x layout is (N, C, D, H, W)
        int n = idx / (channels * depth * height * width);
        int remainder = idx % (channels * depth * height * width);
        int c = remainder / (depth * height * width);
        
        // Load values
        float val = x[idx];
        float scale = scaling_factor[c];
        float b = bias[c];
        
        // Fused operations: scaling, tanh, bias multiplication, sigmoid
        val = val * scale;
        val = tanhf(val);
        val = val * b;
        val = 1.0f / (1.0f + expf(-val));
        
        out[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor scaling_factor, torch::Tensor bias) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    // Get dimensions for indexing
    // x shape: (N, C, D, H, W)
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        scaling_factor.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size, C, D, H, W
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor scaling_factor, torch::Tensor bias);"
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
    Optimized Model that performs a 3D convolution followed by a fused kernel for scaling, tanh, bias multiplication, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # Initialize parameters to match the original model's logic
        # Note: The original model initialized scaling_factor with randn, but the argument 'scaling_factor' passed to init was an int.
        # To match the original behavior strictly, we treat the argument 'scaling_factor' as the shape or value?
        # Looking at original code: self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        # The argument 'scaling_factor' in __init__ was ignored in the original code provided in the prompt description?
        # Wait, the prompt says: def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        # self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        # It seems the argument 'scaling_factor' (which is 2 in the example) is unused in the original __init__ logic provided.
        # It creates a random tensor of shape bias_shape.
        # I will replicate this behavior.
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # Apply the fused CUDA kernel
        x = self.fused_ops.fused_ops_cuda(x, self.scaling_factor, self.bias)
        return x