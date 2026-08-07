import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LeakyReLU and GELU
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_leaky_gelu_kernel(const float* __restrict__ x, float* __restrict__ out, int size, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // LeakyReLU
        val = val > 0 ? val : val * negative_slope;
        
        // GELU (approximation)
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Using tanhf for float precision
        float cdf = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        out[idx] = cdf;
    }
}

torch::Tensor fused_leaky_gelu_cuda(torch::Tensor x, float negative_slope) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_leaky_gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, negative_slope);
    
    return out;
}
"""

fused_activation_cpp_source = (
    "torch::Tensor fused_leaky_gelu_cuda(torch::Tensor x, float negative_slope);"
)

# Compile the inline CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_leaky_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, multiplies by a learnable scalar, 
    and applies a fused LeakyReLU + GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        # Store the negative slope for LeakyReLU (default is 0.01)
        self.negative_slope = 0.01
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        # Apply fused LeakyReLU + GELU
        x = self.fused_activation.fused_leaky_gelu_cuda(x, self.negative_slope)
        return x