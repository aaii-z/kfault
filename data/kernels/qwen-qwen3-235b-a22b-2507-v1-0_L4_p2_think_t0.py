import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

bias_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_clamp_kernel(
    const float* input,
    const float* bias,
    float* output,
    int total_elements,
    int out_channels,
    int height,
    int width,
    float upper_bound
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int spatial_size = height * width;
    int c = (idx / spatial_size) % out_channels;

    float val = input[idx] + bias[c];
    val = fmaxf(0.0f, fminf(val, upper_bound));
    output[idx] = val;
}

torch::Tensor bias_clamp_cuda(torch::Tensor input, torch::Tensor bias, float upper_bound) {
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    int batch_size = input.size(0);
    int out_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    bias_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        out_channels,
        height,
        width,
        upper_bound
    );

    return output;
}
"""

bias_clamp_cpp = "torch::Tensor bias_clamp_cuda(torch::Tensor input, torch::Tensor bias, float upper_bound);"

bias_clamp = load_inline(
    name="bias_clamp",
    cpp_sources=bias_clamp_cpp,
    cuda_sources=bias_clamp_source,
    functions=["bias_clamp_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.bias_clamp = bias_clamp

    def forward(self, x):
        x = self.conv_transpose(x)
        upper_bound = 1.0 / self.scaling_factor
        x = self.bias_clamp.bias_clamp_cuda(x, self.bias, upper_bound)
        return x