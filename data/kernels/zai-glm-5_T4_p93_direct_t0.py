import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Approximation for GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_ops_kernel(const float* __restrict__ x, float* __restrict__ out, int size, float add_value, float multiply_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] + add_value;
        val = fminf(val, 0.0f);
        val = gelu_approx(val);
        out[idx] = val * multiply_value;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, float add_value, float multiply_value) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, add_value, multiply_value);

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, float add_value, float multiply_value);"
)

# Compile the inline CUDA code for fused operations
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution followed by a fused CUDA kernel.
    The fused kernel combines addition, minimum, GELU, and multiplication into a single operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.add_value, self.multiply_value)
        return x