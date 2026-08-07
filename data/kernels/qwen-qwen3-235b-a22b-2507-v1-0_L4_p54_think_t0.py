import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_kernel(const float* conv_out, const float* multiplier, float* out, 
                               int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * height * width;
    if (idx >= total_elements) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (width * height * channels);

    float x = conv_out[idx];
    float m = multiplier[c];
    x *= m;
    
    // LeakyReLU with negative_slope=0.01
    x = (x > 0.0f) ? x : 0.01f * x;
    
    // GELU computation: x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
    float gelu_out = x * 0.5f * (1.0f + erf(x * 0.70710678118f));
    
    out[idx] = gelu_out;
}

torch::Tensor fused_op_cuda(torch::Tensor conv_out, torch::Tensor multiplier) {
    auto batch = conv_out.size(0);
    auto channels = conv_out.size(1);
    auto height = conv_out.size(2);
    auto width = conv_out.size(3);
    
    auto out = torch::empty_like(conv_out);
    int total_elements = batch * channels * height * width;
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_op_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        out.data_ptr<float>(),
        batch, channels, height, width
    );
    
    return out;
}
"""

fused_op_cpp = "torch::Tensor fused_op_cuda(torch::Tensor conv_out, torch::Tensor multiplier);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses multiplication, LeakyReLU, and GELU into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_op_cuda(x, self.multiplier)
        return x