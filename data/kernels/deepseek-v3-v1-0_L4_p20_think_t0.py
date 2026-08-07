import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA operations
custom_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_bias_residual_kernel(
    const float* input,
    const float* bias,
    float* output,
    int total_elements,
    int bias_size,
    int channel_stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int channel_idx = (idx / channel_stride) % bias_size;
        float bias_val = bias[channel_idx];
        float input_val = input[idx];
        float result = input_val + bias_val;  // Add bias
        result += input_val;                  // First residual add
        result *= input_val;                  // Multiply with original
        result += input_val;                  // Second residual add
        output[idx] = result;
    }
}

torch::Tensor fused_bias_residual_cuda(
    torch::Tensor input,
    torch::Tensor bias
) {
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    int bias_size = bias.size(0);
    int channel_stride = input.numel() / input.size(0) / bias_size;
    
    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_bias_residual_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        bias_size,
        channel_stride
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor fused_bias_residual_cuda(torch::Tensor input, torch::Tensor bias);"

# Load the custom CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=custom_ops_source,
    functions=['fused_bias_residual_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ops.fused_bias_residual_cuda(x, self.bias)
        return x