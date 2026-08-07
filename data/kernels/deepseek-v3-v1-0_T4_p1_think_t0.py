import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ReLU and bias addition
fused_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_bias_kernel(const float* input, const float* bias, float* output, int total_elements, int channels, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int c = (index % (channels * height * width)) / (height * width);
        float value = input[index];
        value = fmaxf(0.0f, value) + bias[c];
        output[index] = value;
    }
}

torch::Tensor fused_relu_bias_cuda(torch::Tensor input, torch::Tensor bias) {
    auto output = torch::zeros_like(input);
    int total_elements = input.numel();
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_relu_bias_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), total_elements, channels, height, width);

    return output;
}
"""

fused_relu_bias_cpp_source = "torch::Tensor fused_relu_bias_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the inline CUDA code
fused_relu_bias = load_inline(
    name="fused_relu_bias",
    cpp_sources=fused_relu_bias_cpp_source,
    cuda_sources=fused_relu_bias_source,
    functions=["fused_relu_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_relu_bias = fused_relu_bias

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_relu_bias.fused_relu_bias_cuda(x, self.bias)
        return x