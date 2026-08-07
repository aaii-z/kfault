import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for tanh, scaling, and bias addition
fused_tanh_scale_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

__global__ void fused_tanh_scale_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    float scaling_factor,
    int num_elements,
    int channels,
    int hw_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int channel_idx = (idx / hw_size) % channels;
        float val = input[idx];
        val = tanhf(val);
        val = val * scaling_factor + bias[channel_idx];
        output[idx] = val;
    }
}

torch::Tensor fused_tanh_scale_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor
) {
    auto sizes = input.sizes();
    int num_elements = input.numel();
    int channels = sizes[1];
    int hw_size = sizes[2] * sizes[3];
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_tanh_scale_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        num_elements,
        channels,
        hw_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_tanh_scale_bias_cpp_source = "torch::Tensor fused_tanh_scale_bias_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_tanh_scale_bias_cpp_source,
    cuda_sources=fused_tanh_scale_bias_source,
    functions=["fused_tanh_scale_bias_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_tanh_scale_bias_cuda(x, self.bias, self.scaling_factor)
        x = self.max_pool(x)
        return x