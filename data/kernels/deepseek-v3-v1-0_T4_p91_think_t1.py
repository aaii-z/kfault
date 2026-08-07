import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias add, scaling, and sigmoid
bias_scale_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_scale_sigmoid_kernel(
    const float* input,
    const float* bias,
    float scaling_factor,
    float* output,
    int num_elements,
    int num_channels,
    int spatial_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;
    
    int channel = (index / spatial_size) % num_channels;
    float value = input[index] + bias[channel];
    value = value * scaling_factor;
    output[index] = 1.0f / (1.0f + expf(-value));
}

torch::Tensor bias_scale_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor
) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    int num_channels = input.size(1);
    int spatial_size = input.size(2) * input.size(3);
    
    const int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    
    bias_scale_sigmoid_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        num_elements,
        num_channels,
        spatial_size
    );
    
    return output;
}
"""

bias_scale_sigmoid_cpp_source = """
torch::Tensor bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);
"""

# Compile the inline CUDA code
bias_scale_sigmoid = load_inline(
    name="bias_scale_sigmoid",
    cpp_sources=bias_scale_sigmoid_cpp_source,
    cuda_sources=bias_scale_sigmoid_source,
    functions=["bias_scale_sigmoid_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_op = bias_scale_sigmoid

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.softmax(x, dim=1)
        x = self.fused_op.bias_scale_sigmoid_cuda(x, self.bias, self.scaling_factor)
        return x