import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bn_scale_kernel(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    int size,
    int channels,
    int channel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = idx / channel_size;
        float s = scale[c];
        float b = bias[c];
        output[idx] = input[idx] * s + b;
    }
}

torch::Tensor fused_bn_scale_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {
    int size = input.numel();
    int channels = scale.size(0);
    int channel_size = size / channels;

    auto output = torch::empty_like(input);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_bn_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        channels,
        channel_size
    );

    return output;
}
"""

fused_bn_scale_cpp = "torch::Tensor fused_bn_scale_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);"

fused_bn_scale = load_inline(
    name="fused_bn_scale",
    cpp_sources=fused_bn_scale_cpp,
    cuda_sources=fused_bn_scale_source,
    functions=["fused_bn_scale_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fused_bn_scale_op = fused_bn_scale

    def forward(self, x):
        x = self.conv(x)
        eps = self.bn.eps
        gamma = self.bn.weight
        beta = self.bn.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        
        std = torch.sqrt(running_var + eps)
        fused_scale = self.scaling_factor * gamma / std
        fused_bias = self.scaling_factor * (beta - running_mean * gamma / std)
        
        x = self.fused_bn_scale_op.fused_bn_scale_cuda(x, fused_scale, fused_bias)
        return x