import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_kernel(const float* input, const float* sum_tensor, float* output, 
                            int batch, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * depth * height * width;
    if (idx >= total_elements) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % channels;
    int n = idx / (width * height * depth * channels);

    float x = input[idx];

    // LeakyReLU with negative_slope=0.2
    x = (x > 0.0f) ? x : 0.2f * x;

    // Add sum_tensor[c]
    x += sum_tensor[c];

    // Clamp to [-1.0, 1.0]
    x = (x < -1.0f) ? -1.0f : (x > 1.0f) ? 1.0f : x;

    // GELU using erf approximation
    float x_scaled = x * 0.70710678118f;
    float cdf = 0.5f * (1.0f + erff(x_scaled));
    x = x * cdf;

    output[idx] = x;
}

torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor sum_tensor) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto output = torch::empty_like(input);
    int total_elements = batch * channels * depth * height * width;

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        sum_tensor.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch, channels, depth, height, width
    );

    return output;
}
"""

fused_op_cpp = "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor sum_tensor);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_op_cuda(x, self.sum_tensor.view(-1))
        return x