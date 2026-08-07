import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-convolution operations
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(const float* __restrict__ y, const float* __restrict__ bias, float* __restrict__ out, int size, int C, int stride_c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Calculate channel index based on the stride of the channel dimension
        // Assuming y is contiguous (N, C, D, H, W)
        int c = (idx / stride_c) % C;
        
        float y_val = y[idx];
        float b_val = bias[c];
        
        // Original sequence:
        // x = y + b
        // x = x + y -> 2y + b
        // x = x * y -> 2y^2 + by
        // x = x + y -> 2y^2 + by + y
        // Simplified: y * (2y + b + 1)
        
        out[idx] = y_val * (2.0f * y_val + b_val + 1.0f);
    }
}

torch::Tensor fused_post_conv_cuda(torch::Tensor y, torch::Tensor bias) {
    auto size = y.numel();
    auto out = torch::empty_like(y);
    
    int C = y.size(1);
    // Stride for the channel dimension (D * H * W)
    int stride_c = y.stride(1);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_post_conv_kernel<<<num_blocks, block_size>>>(
        y.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), size, C, stride_c
    );
    
    return out;
}
"""

fused_post_conv_cpp_source = (
    "torch::Tensor fused_post_conv_cuda(torch::Tensor y, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_post_conv = load_inline(
    name="fused_post_conv",
    cpp_sources=fused_post_conv_cpp_source,
    cuda_sources=fused_post_conv_source,
    functions=["fused_post_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    Optimized with a custom CUDA kernel that fuses the post-convolution operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_post_conv = fused_post_conv

    def forward(self, x):
        x = self.conv_transpose(x)
        # The custom kernel fuses: x + bias, x + original_x, x * original_x, x + original_x
        # Note: This operation does not support autograd for the custom kernel in this basic implementation,
        # matching the style of the provided example.
        x = self.fused_post_conv.fused_post_conv_cuda(x, self.bias)
        return x