import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_multiply_leaky_relu_kernel(
    const float* input,
    const float* multiplier,
    float* output,
    int total_elements,
    int channels,
    int d, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int spatial_size = d * h * w;
    int c = (idx / spatial_size) % channels;
    float m = multiplier[c];
    float x = input[idx] * m;
    output[idx] = (x > 0) ? x : 0.2f * x;
}

torch::Tensor fused_multiply_leaky_relu_cuda(torch::Tensor input, torch::Tensor multiplier) {
    auto sizes = input.sizes();
    int64_t batch = sizes[0];
    int64_t channels = sizes[1];
    int64_t d = sizes[2];
    int64_t h = sizes[3];
    int64_t w = sizes[4];
    int64_t total_elements = input.numel();

    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    int d_int = static_cast<int>(d);
    int h_int = static_cast<int>(h);
    int w_int = static_cast<int>(w);
    int channels_int = static_cast<int>(channels);

    fused_multiply_leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        channels_int,
        d_int, h_int, w_int
    );

    return output;
}
"""

fused_cpp_source = "torch::Tensor fused_multiply_leaky_relu_cuda(torch::Tensor input, torch::Tensor multiplier);"

fused_op = load_inline(
    name="fused_multiply_leaky_relu",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_multiply_leaky_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.leaky_relu(x)
        x = self.fused_op.fused_multiply_leaky_relu_cuda(x, self.multiplier)
        x = self.max_pool(x)
        return x