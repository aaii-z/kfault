import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_bias_kernel(const float* y, float* out, const float* bias, int size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / (height * width)) % channels;
        float y_val = y[idx];
        float b = bias[c];
        out[idx] = (y_val > 0.0f) ? (y_val + b) : b;
    }
}

torch::Tensor fused_relu_bias_cuda(torch::Tensor y, torch::Tensor bias) {
    auto size = y.numel();
    auto out = torch::empty_like(y);

    int channels = y.size(1);
    int height = y.size(2);
    int width = y.size(3);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_relu_bias_kernel<<<num_blocks, block_size>>>(y.data_ptr<float>(), out.data_ptr<float>(), bias.data_ptr<float>(), size, channels, height, width);

    return out;
}
"""

fused_relu_bias_cpp_source = (
    "torch::Tensor fused_relu_bias_cuda(torch::Tensor y, torch::Tensor bias);"
)

fused_relu_bias = load_inline(
    name="fused_relu_bias",
    cpp_sources=fused_relu_bias_cpp_source,
    cuda_sources=fused_relu_bias_source,
    functions=["fused_relu_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_relu_bias = fused_relu_bias

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_relu_bias.fused_relu_bias_cuda(x, self.bias)
        return x