import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min reduction and double tanh
min_tanh2_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void min_tanh2_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int spatial_size = height * width;
    int total_elements = batch_size * spatial_size;
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    int b = index / spatial_size;
    int s = index % spatial_size;
    int h = s / width;
    int w = s % width;

    const float* channel_start = input + (b * channels * spatial_size) + s;
    float min_val = FLT_MAX;

    for (int c = 0; c < channels; ++c) {
        float val = channel_start[c * spatial_size];
        if (val < min_val) min_val = val;
    }

    float temp = tanhf(min_val);
    float result = tanhf(temp);
    output[b * spatial_size + s] = result;
}

torch::Tensor min_tanh2_cuda(
    torch::Tensor input,
    int channels
) {
    int batch_size = input.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int spatial_size = height * width;
    int total_elements = batch_size * spatial_size;

    auto output = torch::empty({batch_size, 1, height, width}, input.options());

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    min_tanh2_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );

    return output;
}
"""

min_tanh2_cpp = "torch::Tensor min_tanh2_cuda(torch::Tensor input, int channels);"

# Load the custom CUDA extension
min_tanh2_ext = load_inline(
    name="min_tanh2",
    cpp_sources=min_tanh2_cpp,
    cuda_sources=min_tanh2_kernel,
    functions=["min_tanh2_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        x = min_tanh2_ext.min_tanh2_cuda(x, self.out_channels)
        return x