import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused subtraction, tanh, and subtraction
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_sub_tanh_sub_kernel(const float* in, float* out, int size, float s1, float s2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = in[idx] - s1;
        val = tanhf(val);
        out[idx] = val - s2;
    }
}

torch::Tensor fused_sub_tanh_sub_cuda(torch::Tensor in, float s1, float s2) {
    auto size = in.numel();
    auto out = torch::empty_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_sub_tanh_sub_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), size, s1, s2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_ops_cpp = "torch::Tensor fused_sub_tanh_sub_cuda(torch::Tensor in, float s1, float s2);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_ops_source,
    functions=["fused_sub_tanh_sub_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for subtraction, tanh, and subtraction operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.fused_op = fused_ops.fused_sub_tanh_sub_cuda

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op(x, self.subtract1_value, self.subtract2_value)
        x = self.avgpool(x)
        return x