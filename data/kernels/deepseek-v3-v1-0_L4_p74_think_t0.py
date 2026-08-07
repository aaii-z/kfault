import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: LeakyReLU + Multiply + LeakyReLU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_leaky_multiply_leaky_kernel(
    const float* input,
    const float* multiplier,
    float* output,
    int num_elements,
    int channels,
    int spatial_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        int c = (index / spatial_size) % channels;
        float x_val = input[index];
        float leaky1 = (x_val > 0) ? x_val : 0.2f * x_val;
        float scaled = leaky1 * multiplier[c];
        output[index] = (scaled > 0) ? scaled : 0.2f * scaled;
    }
}

torch::Tensor fused_leaky_multiply_leaky_cuda(
    torch::Tensor input,
    torch::Tensor multiplier
) {
    auto num_elements = input.numel();
    auto output = torch::empty_like(input);
    int channels = input.size(1);
    int spatial_size = input.size(2) * input.size(3) * input.size(4);

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_leaky_multiply_leaky_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        channels,
        spatial_size
    );

    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_leaky_multiply_leaky_cuda(torch::Tensor input, torch::Tensor multiplier);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_leaky_multiply_leaky_cuda"],
    verbose=True,
    extra_cflags=[],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_leaky_multiply_leaky_cuda(x, self.multiplier)
        x = self.max_pool(x)
        return x