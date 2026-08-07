import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_add_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_hardswish_kernel(const float* x_conv, const float* add_input, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = x_conv[idx] + add_input[idx];
        float z = y + 3.0f;
        z = (z < 0.0f) ? 0.0f : (z > 6.0f) ? 6.0f : z;
        out[idx] = y * y * z / 6.0f;
    }
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor x_conv, torch::Tensor add_input) {
    TORCH_CHECK(x_conv.device().is_cuda(), "x_conv must be a CUDA tensor");
    TORCH_CHECK(add_input.device().is_cuda(), "add_input must be a CUDA tensor");
    TORCH_CHECK(x_conv.dtype() == torch::kFloat32, "x_conv must be float32");
    TORCH_CHECK(add_input.dtype() == torch::kFloat32, "add_input must be float32");
    TORCH_CHECK(x_conv.sizes() == add_input.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(x_conv.is_contiguous(), "x_conv must be contiguous");
    TORCH_CHECK(add_input.is_contiguous(), "add_input must be contiguous");

    auto out = torch::empty_like(x_conv);
    int size = x_conv.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_hardswish_kernel<<<num_blocks, block_size>>>(
        x_conv.data_ptr<float>(),
        add_input.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );

    return out;
}
"""

fused_add_hardswish_cpp = "torch::Tensor fused_add_hardswish_cuda(torch::Tensor x_conv, torch::Tensor add_input);"

fused_add_hardswish = load_inline(
    name="fused_add_hardswish",
    cpp_sources=fused_add_hardswish_cpp,
    cuda_sources=fused_add_hardswish_source,
    functions=["fused_add_hardswish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses element-wise addition and HardSwish activation into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_add_hardswish

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution, of shape (batch_size, out_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W) after fused operation.
        """
        x = self.conv_transpose(x)
        return self.fused_op.fused_add_hardswish_cuda(x, add_input)