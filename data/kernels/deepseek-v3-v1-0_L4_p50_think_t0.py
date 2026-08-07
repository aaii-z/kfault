import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias addition and scaling
fused_bias_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_scale_kernel(const float* input, const float* bias, float scale, float* output, int num_elements, int spatial_size, int num_channels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        int c = (index / spatial_size) % num_channels;
        output[index] = (input[index] + bias[c]) * scale;
    }
}

torch::Tensor fused_bias_scale_cuda(torch::Tensor input, torch::Tensor bias, float scale) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    int spatial_size = input.size(2) * input.size(3) * input.size(4);
    int num_channels = input.size(1);

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_bias_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale,
        output.data_ptr<float>(),
        num_elements,
        spatial_size,
        num_channels
    );

    return output;
}
"""

fused_bias_scale_cpp_source = "torch::Tensor fused_bias_scale_cuda(torch::Tensor input, torch::Tensor bias, float scale);"

# Compile the inline CUDA code
fused_bias_scale = load_inline(
    name="fused_bias_scale",
    cpp_sources=fused_bias_scale_cpp_source,
    cuda_sources=fused_bias_scale_source,
    functions=["fused_bias_scale_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        
        # Fuse scale1 into conv weights during initialization
        temp_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        with torch.no_grad():
            self.conv_transpose.weight.copy_(temp_conv.weight * scale1)
            conv_bias = temp_conv.bias
            total_bias = (conv_bias * scale1) + torch.randn(bias_shape)
        
        self.total_bias = nn.Parameter(total_bias)
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.fused_bias_scale = fused_bias_scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.avg_pool(x)
        x = self.fused_bias_scale.fused_bias_scale_cuda(x, self.total_bias.squeeze(), self.scale2.item())
        return x