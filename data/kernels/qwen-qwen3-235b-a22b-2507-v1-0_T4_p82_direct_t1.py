import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Tanh + Scale + Bias
tanh_scale_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_scale_bias_kernel(const float* x, float* out, const float* bias, float scale, int size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / spatial_size) % channels;
        out[idx] = tanhf(x[idx]) * scale + bias[c];
    }
}

torch::Tensor tanh_scale_bias_cuda(torch::Tensor x, torch::Tensor bias, float scale) {
    auto size = x.numel();
    auto channels = bias.size(0);
    auto spatial_size = x.size(2) * x.size(3);
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_scale_bias_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale,
        size,
        channels,
        spatial_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

tanh_scale_bias_cpp = """
torch::Tensor tanh_scale_bias_cuda(torch::Tensor x, torch::Tensor bias, float scale);
"""

tanh_scale_bias_module = load_inline(
    name="tanh_scale_bias",
    cpp_sources=tanh_scale_bias_cpp,
    cuda_sources=tanh_scale_bias_source,
    functions=["tanh_scale_bias_cuda"],
    verbose=False,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.tanh_scale_bias_op = tanh_scale_bias_module.tanh_scale_bias_cuda

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh_scale_bias_op(x, self.bias, self.scaling_factor)
        x = self.max_pool(x)
        return x