import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused InstanceNorm2d + Division CUDA kernel
fused_norm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void instance_norm_div_kernel(float* data, int batch_size, int channels, int height_width, float divide_by, float epsilon) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int idx = (n * channels + c) * height_width;
    
    // Calculate mean and variance for this instance and channel
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = 0; i < height_width; i++) {
        float val = data[idx + i];
        sum += val;
        sum_sq += val * val;
    }
    
    float mean = sum / height_width;
    float variance = (sum_sq / height_width) - (mean * mean);
    
    // Apply instance normalization and division
    for (int i = 0; i < height_width; i++) {
        int pos = idx + i;
        float normalized = (data[pos] - mean) / sqrtf(variance + epsilon);
        data[pos] = normalized / divide_by;
    }
}

torch::Tensor fused_norm_div_forward(torch::Tensor input, float divide_by) {
    CHECK_INPUT(input);
    
    auto output = input.clone();
    int batch_size = output.size(0);
    int channels = output.size(1);
    int height = output.size(2);
    int width = output.size(3);
    int height_width = height * width;
    
    float epsilon = 1e-5f;
    
    dim3 grid(batch_size, channels);
    instance_norm_div_kernel<<<grid, 1>>>(output.data_ptr<float>(), batch_size, channels, height_width, divide_by, epsilon);
    
    return output;
}
"""

fused_norm_div_cpp_source = """
torch::Tensor fused_norm_div_forward(torch::Tensor input, float divide_by);
"""

# Compile the inline CUDA code
fused_norm_div = load_inline(
    name="fused_norm_div",
    cpp_sources=fused_norm_div_cpp_source,
    cuda_sources=fused_norm_div_source,
    functions=["fused_norm_div_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.divide_by = divide_by
        self.fused_norm_div = fused_norm_div

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_norm_div.fused_norm_div_forward(x, self.divide_by)
        return x