import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

bias_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_add_kernel(const float* input, const float* bias, float* output, int num_elements, int spatial_size, int num_channels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        int c = (index / spatial_size) % num_channels;
        output[index] = input[index] + bias[c];
    }
}

torch::Tensor bias_add_cuda(torch::Tensor input, torch::Tensor bias) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    int spatial_size = input.size(2) * input.size(3) * input.size(4);
    int num_channels = input.size(1);

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    bias_add_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        spatial_size,
        num_channels
    );

    return output;
}
"""

bias_add_cpp_source = "torch::Tensor bias_add_cuda(torch::Tensor input, torch::Tensor bias);"

bias_add = load_inline(
    name="bias_add",
    cpp_sources=bias_add_cpp_source,
    cuda_sources=bias_add_source,
    functions=["bias_add_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.bias_add = bias_add

        with torch.no_grad():
            self.conv_transpose.weight.mul_(scale1)
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.mul_(scale1)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.avg_pool(x)
        x = self.bias_add.bias_add_cuda(x, self.bias.squeeze())
        x = x * self.scale2
        return x