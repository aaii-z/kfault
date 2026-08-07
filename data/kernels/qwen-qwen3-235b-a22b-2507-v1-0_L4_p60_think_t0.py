import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void swish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val * (1.0f / (1.0f + expf(-val)));
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""
swish_cpp = "torch::Tensor swish_cuda(torch::Tensor x);"

swish = load_inline(
    name="swish",
    cpp_sources=swish_cpp,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardswish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float temp = val + 3.0f;
        temp = (temp > 6.0f) ? 6.0f : temp;
        temp = (temp < 0.0f) ? 0.0f : temp;
        out[idx] = val * temp / 6.0f;
    }
}

torch::Tensor hardswish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    hardswish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""
hardswish_cpp = "torch::Tensor hardswish_cuda(torch::Tensor x);"

hardswish = load_inline(
    name="hardswish",
    cpp_sources=hardswish_cpp,
    cuda_sources=hardswish_source,
    functions=["hardswish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.swish = swish
        self.hardswish = hardswish

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.swish.swish_cuda(x)
        x = self.group_norm(x)
        x = self.hardswish.hardswish_cuda(x)
        return x