import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_tanh_scale_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_tanh_scale_bias_kernel(
    const float* input,
    float* output,
    float scaling_factor,
    const float* bias,
    int batch_size,
    int out_channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    if (idx < total_elements) {
        int c = (idx / (height * width)) % out_channels;
        float x = input[idx];
        float tanh_x = tanhf(x);
        output[idx] = tanh_x * scaling_factor + bias[c];
    }
}

torch::Tensor fused_tanh_scale_bias_cuda(
    torch::Tensor input,
    float scaling_factor,
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto out_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_tanh_scale_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        bias.data_ptr<float>(),
        batch_size,
        out_channels,
        height,
        width
    );
    
    return output;
}
"""

fused_tanh_scale_bias_cpp = "torch::Tensor fused_tanh_scale_bias_cuda(torch::Tensor input, float scaling_factor, torch::Tensor bias);"

fused_tanh_scale_bias = load_inline(
    name="fused_tanh_scale_bias",
    cpp_sources=fused_tanh_scale_bias_cpp,
    cuda_sources=fused_tanh_scale_bias_source,
    functions=["fused_tanh_scale_bias_cuda"],
    verbose=False,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused tanh, scaling, and bias addition operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.fused_op = fused_tanh_scale_bias

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_tanh_scale_bias_cuda(x, self.scaling_factor, self.bias)
        x = self.max_pool(x)
        return x