import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for bias subtraction and tanh activation
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void bias_subtract_tanh_kernel(
    const float* input,
    const float* bias,
    float* output,
    int total_elements,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int c = (idx / (height * width)) % channels;
    float bias_val = bias[c];
    float val = input[idx] - bias_val;
    output[idx] = tanhf(val);
}

torch::Tensor fused_bias_subtract_tanh_cuda(torch::Tensor input, torch::Tensor bias) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int height = sizes[2];
    int width = sizes[3];
    
    int total_elements = batch_size * channels * height * width;
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    bias_subtract_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        channels,
        height,
        width
    );
    
    return output;
}
"""

fused_kernel_cpp = "torch::Tensor fused_bias_subtract_tanh_cuda(torch::Tensor input, torch::Tensor bias);"

# Load the custom CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=['fused_bias_subtract_tanh_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ops.fused_bias_subtract_tanh_cuda(x, self.bias)
        return x