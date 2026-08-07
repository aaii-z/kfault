import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused min, add, and scale operations
min_add_scale_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void min_add_scale_kernel(
    const float* input,
    const float* bias,
    float constant_value,
    float scaling_factor,
    float* output,
    int total_elements,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int c = (idx / spatial_size) % channels;
        float val = input[idx];
        val = fminf(val, constant_value);
        val += bias[c];
        val *= scaling_factor;
        output[idx] = val;
    }
}

torch::Tensor min_add_scale_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float constant_value,
    float scaling_factor
) {
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    int channels = input.size(1);
    int spatial_size = input.size(2) * input.size(3);

    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    min_add_scale_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        constant_value,
        scaling_factor,
        output.data_ptr<float>(),
        total_elements,
        channels,
        spatial_size
    );

    return output;
}
"""

min_add_scale_cpp = "torch::Tensor min_add_scale_cuda(torch::Tensor input, torch::Tensor bias, float constant_value, float scaling_factor);"

min_add_scale = load_inline(
    name="min_add_scale",
    cpp_sources=min_add_scale_cpp,
    cuda_sources=min_add_scale_source,
    functions=["min_add_scale_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.min_add_scale = min_add_scale

    def forward(self, x):
        x = self.conv(x)
        x = self.min_add_scale.min_add_scale_cuda(
            x, 
            self.bias, 
            self.constant_value, 
            self.scaling_factor
        )
        return x