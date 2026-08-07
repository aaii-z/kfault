import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused add and HardSwish
fused_add_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_hardswish_kernel(const float* x, const float* add_input, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] + add_input[idx];
        // HardSwish: x * relu6(x + 3) / 6
        // relu6(x + 3) is equivalent to min(max(x + 3, 0), 6)
        float temp = val + 3.0f;
        if (temp < 0.0f) temp = 0.0f;
        else if (temp > 6.0f) temp = 6.0f;
        out[idx] = val * temp / 6.0f;
    }
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add_input) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_hardswish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), add_input.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_add_hardswish_cpp_source = (
    "torch::Tensor fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add_input);"
)

# Compile the inline CUDA code for fused add and HardSwish
fused_add_hardswish = load_inline(
    name="fused_add_hardswish",
    cpp_sources=fused_add_hardswish_cpp_source,
    cuda_sources=fused_add_hardswish_source,
    functions=["fused_add_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    The addition and HardSwish are fused into a single CUDA kernel for better performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_add_hardswish

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution, of shape (batch_size, out_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W) after HardSwish activation.
        """
        x = self.conv_transpose(x)
        x = x + self.bias
        x = self.fused_op.fused_add_hardswish_cuda(x, add_input)
        return x