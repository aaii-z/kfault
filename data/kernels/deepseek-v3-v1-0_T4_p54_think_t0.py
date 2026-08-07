import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for element-wise multiplication, LeakyReLU, and GELU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_mult_leaky_relu_gelu_kernel(
    const float* input,
    const float* multiplier,
    float* output,
    int num_elements,
    int channels,
    int height_width,
    float negative_slope
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        int channel_idx = (index / height_width) % channels;
        float val = input[index] * multiplier[channel_idx];
        val = val > 0 ? val : val * negative_slope;
        output[index] = gelu_approx(val);
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor multiplier,
    float negative_slope
) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    int channels = input.size(1);
    int height_width = input.size(2) * input.size(3);

    const int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    fused_mult_leaky_relu_gelu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        channels,
        height_width,
        negative_slope
    );

    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor multiplier, float negative_slope);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.negative_slope = 0.01  # Default LeakyReLU negative slope

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops.fused_ops_cuda(x, self.multiplier, self.negative_slope)
        return x