import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Mish and Tanh
fused_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_mish_tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = input[idx];
        float exp_x = expf(x_val);
        float softplus = log1pf(exp_x);
        float tanh_softplus = tanhf(softplus);
        float mish_val = x_val * tanh_softplus;
        float result = tanhf(mish_val);
        output[idx] = result;
    }
}

torch::Tensor fused_mish_tanh_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    fused_mish_tanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

fused_mish_tanh_cpp_source = "torch::Tensor fused_mish_tanh_cuda(torch::Tensor input);"

# Compile the inline CUDA code for fused Mish and Tanh
fused_mish_tanh = load_inline(
    name="fused_mish_tanh",
    cpp_sources=fused_mish_tanh_cpp_source,
    cuda_sources=fused_mish_tanh_source,
    functions=["fused_mish_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that replaces the Mish and Tanh activations with a fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_mish_tanh = fused_mish_tanh

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv(x)
        x = self.fused_mish_tanh.fused_mish_tanh_cuda(x)
        return x