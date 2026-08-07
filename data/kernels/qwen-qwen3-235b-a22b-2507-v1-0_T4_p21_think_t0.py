import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bias_scale_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bias_scale_sigmoid_kernel(
    const float* x, 
    const float* bias, 
    const float* scale, 
    float* out, 
    int batch_size, 
    int channels, 
    int height, 
    int width) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    if (idx >= total_elements) return;

    int spatial_size = height * width;
    int c = (idx / spatial_size) % channels;

    float x_val = x[idx];
    float bias_val = bias[c];
    float scale_val = scale[c];
    float temp = (x_val + bias_val) * scale_val;
    out[idx] = 1.0f / (1.0f + expf(-temp));
}

torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor x, 
    torch::Tensor bias, 
    torch::Tensor scale) 
{
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out = torch::empty_like(x);

    const int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        bias.reshape({channels}).data_ptr<float>(), 
        scale.reshape({channels}).data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, 
        channels, 
        height, 
        width
    );

    return out;
}
"""

fused_bias_scale_sigmoid_cpp = "torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor x, torch::Tensor bias, torch::Tensor scale);"

fused_bias_scale_sigmoid = load_inline(
    name="fused_bias_scale_sigmoid",
    cpp_sources=fused_bias_scale_sigmoid_cpp,
    cuda_sources=fused_bias_scale_sigmoid_source,
    functions=["fused_bias_scale_sigmoid_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_op = fused_bias_scale_sigmoid

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_bias_scale_sigmoid_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x