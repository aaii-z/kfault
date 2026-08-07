import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Division and LeakyReLU
# We keep the Conv2d as a standard PyTorch operation (leveraging cuDNN) and fuse the subsequent element-wise ops.
fused_div_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void div_leaky_relu_kernel(const float* __restrict__ input, float* __restrict__ output, int size, float inv_divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float negative_slope = 0.01f;
    if (idx < size) {
        float val = input[idx] * inv_divisor;
        output[idx] = val > 0.0f ? val : val * negative_slope;
    }
}

torch::Tensor fused_div_leaky_relu_cuda(torch::Tensor input, float divisor) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    float inv_divisor = 1.0f / divisor;

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    div_leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, 
        inv_divisor
    );

    return output;
}
"""

fused_div_leaky_relu_cpp_source = "torch::Tensor fused_div_leaky_relu_cuda(torch::Tensor input, float divisor);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_div_leaky_relu",
    cpp_sources=fused_div_leaky_relu_cpp_source,
    cuda_sources=fused_div_leaky_relu_source,
    functions=["fused_div_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for Division + LeakyReLU fusion.
    The Conv2d operation is performed using the standard PyTorch implementation,
    followed by the custom fused kernel for element-wise operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.fused_op = fused_op

    def forward(self, x):
        # Standard Conv2d
        x = self.conv(x)
        # Fused Division + LeakyReLU
        return self.fused_op.fused_div_leaky_relu_cuda(x, self.divisor)