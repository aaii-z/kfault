import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused subtract and Mish activation
# Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
# We fuse the two subtractions and the Mish activation into a single kernel.
fused_subtract_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_subtract_mish_kernel(const float* __restrict__ input, float* __restrict__ output, int size, float sub1, float sub2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - sub1 - sub2;
        // Compute Mish activation
        // softplus(x) = ln(1 + exp(x))
        // mish(x) = x * tanh(softplus(x))
        float sp = logf(1.0f + expf(x));
        float tanh_sp = tanhf(sp);
        output[idx] = x * tanh_sp;
    }
}

torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float subtract_value_1, float subtract_value_2) {
    auto size = input.numel();
    auto out = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_subtract_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), out.data_ptr<float>(), size, subtract_value_1, subtract_value_2
    );

    return out;
}
"""

fused_subtract_mish_cpp_source = (
    "torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float subtract_value_1, float subtract_value_2);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_subtract_mish",
    cpp_sources=fused_subtract_mish_cpp_source,
    cuda_sources=fused_subtract_mish_source,
    functions=["fused_subtract_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, followed by a fused subtract and Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_subtract_mish_cuda(x, self.subtract_value_1, self.subtract_value_2)
        return x