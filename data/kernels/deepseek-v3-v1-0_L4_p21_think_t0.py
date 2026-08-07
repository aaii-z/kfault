import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for bias addition, scaling, and sigmoid
fused_bias_scale_sigmoid_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_bias_scale_sigmoid_kernel(
    const float* input,
    const float* bias,
    const float* scale,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (index < total_elements) {
        int c = (index / (height * width)) % channels;
        float value = input[index] + bias[c];
        value *= scale[c];
        output[index] = 1.0f / (1.0f + expf(-value));
    }
}

torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor scale
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::empty_like(input);
    
    int total_elements = batch_size * channels * height * width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_bias_scale_sigmoid_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}
"""

fused_bias_scale_sigmoid_cpp_source = """
torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bias_scale_sigmoid_cpp_source,
    cuda_sources=fused_bias_scale_sigmoid_source,
    functions=["fused_bias_scale_sigmoid_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_bias_scale_sigmoid_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x