import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused add and hardswish multiplication
# The operation is: out = (conv_out + add_input) * hardswish(conv_out + add_input)
# hardswish(x) = x * relu6(x + 3) / 6
# So the operation becomes: out = x * (x * relu6(x + 3) / 6) where x = conv_out + add_input

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish_func(float x) {
    // hardswish(x) = x * relu6(x + 3) / 6
    float relu6_val = x + 3.0f;
    if (relu6_val < 0.0f) relu6_val = 0.0f;
    else if (relu6_val > 6.0f) relu6_val = 6.0f;
    return x * relu6_val / 6.0f;
}

__global__ void fused_add_hardswish_mul_kernel(const float* __restrict__ conv_out, 
                                               const float* __restrict__ add_input, 
                                               float* __restrict__ out, 
                                               int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = conv_out[idx] + add_input[idx];
        float hs_val = hardswish_func(val);
        out[idx] = val * hs_val;
    }
}

torch::Tensor fused_add_hardswish_mul_cuda(torch::Tensor conv_out, torch::Tensor add_input) {
    auto size = conv_out.numel();
    auto out = torch::empty_like(conv_out);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_hardswish_mul_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(), 
        add_input.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_add_hardswish_mul_cuda(torch::Tensor conv_out, torch::Tensor add_input);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_add_hardswish_mul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, adds an input tensor, 
    and applies HardSwish activation using a custom fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) # Kept for architecture compatibility, though unused in forward
        self.fused_op = fused_op

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution, of shape (batch_size, out_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W) after HardSwish activation.
        """
        x = self.conv_transpose(x)
        # Fused kernel for: x = x + add_input; x = x * hardswish(x)
        x = self.fused_op.fused_add_hardswish_mul_cuda(x, add_input)
        return x