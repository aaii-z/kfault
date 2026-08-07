import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for scaling, tanh, bias multiplication, and sigmoid
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

__global__ void fused_scale_tanh_bias_sigmoid_kernel(
    const float* input,
    const float* scaling_factor,
    const float* bias,
    float* output,
    int num_elements,
    int channels,
    int depth,
    int height,
    int width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        int n = index / (channels * depth * height * width);
        int c = (index / (depth * height * width)) % channels;
        
        float x = input[index];
        x = x * scaling_factor[c];
        x = tanhf(x);
        x = x * bias[c];
        x = 1.0f / (1.0f + expf(-x));
        output[index] = x;
    }
}

torch::Tensor fused_scale_tanh_bias_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor scaling_factor,
    torch::Tensor bias
) {
    int num_elements = input.numel();
    auto output = torch::empty_like(input);
    
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    fused_scale_tanh_bias_sigmoid_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        scaling_factor.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        channels,
        depth,
        height,
        width
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_scale_tanh_bias_sigmoid_cuda(torch::Tensor input, torch::Tensor scaling_factor, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name='fused_scale_tanh_bias_sigmoid',
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=['fused_scale_tanh_bias_sigmoid_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_scale_tanh_bias_sigmoid_cuda(x, self.scaling_factor, self.bias)
        return x