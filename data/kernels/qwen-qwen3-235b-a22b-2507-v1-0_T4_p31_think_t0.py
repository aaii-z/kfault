import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* conv_out, const float* bias, float* out, 
                            int batch, int channels, int height, int width,
                            float constant_value, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * height * width;
    if (idx >= total_elements) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int b = idx / (width * height * channels);

    int conv_idx = b * channels * height * width + c * height * width + h * width + w;
    float x = conv_out[conv_idx];
    float min_val = (x < constant_value) ? x : constant_value;
    float result = (min_val + bias[c]) * scaling_factor;
    out[conv_idx] = result;
}

torch::Tensor fused_op_cuda(torch::Tensor conv_out, torch::Tensor bias, float constant_value, float scaling_factor) {
    TORCH_CHECK(conv_out.device().is_cuda(), "conv_out must be on CUDA");
    TORCH_CHECK(bias.device().is_cuda(), "bias must be on CUDA");
    TORCH_CHECK(conv_out.dim() == 4, "conv_out must be 4D");
    TORCH_CHECK(bias.dim() == 1 || bias.dim() == 3, "bias must be 1D or 3D");
    TORCH_CHECK(bias.size(0) == conv_out.size(1), "bias channels must match");

    int batch = conv_out.size(0);
    int channels = conv_out.size(1);
    int height = conv_out.size(2);
    int width = conv_out.size(3);

    auto out = torch::empty_like(conv_out);
    int total_elements = batch * channels * height * width;

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch, channels, height, width,
        constant_value, scaling_factor
    );

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor conv_out, torch::Tensor bias, float constant_value, float scaling_factor);"
)

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_op_cuda(x, self.bias, self.constant_value, self.scaling_factor)
        return x