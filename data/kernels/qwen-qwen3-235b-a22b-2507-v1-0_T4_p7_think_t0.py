import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_activation_kernel(
    const float* input,
    const float* bias,
    float* output,
    int B, int C, int D, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * D * H * W;
    if (idx >= total_elements) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int c = (idx / (W * H * D)) % C;

    float x = input[idx];

    // ReLU
    x = (x > 0.0f) ? x : 0.0f;

    // GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    float x_gelu = 0.5f * x * (1.0f + erf(x * 0.70710678118f));

    // Sigmoid
    float x_sigmoid = 1.0f / (1.0f + exp(-x_gelu));

    // Add bias
    output[idx] = x_sigmoid + bias[c];
}

torch::Tensor fused_activation_cuda(torch::Tensor input, torch::Tensor bias) {
    auto B = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto output = torch::empty_like(input);

    const int total_elements = B * C * D * H * W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W
    );

    return output;
}
"""

fused_activation_cpp = "torch::Tensor fused_activation_cuda(torch::Tensor input, torch::Tensor bias);"

fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_activation_op = fused_activation

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation_op.fused_activation_cuda(x, self.bias)
        return x