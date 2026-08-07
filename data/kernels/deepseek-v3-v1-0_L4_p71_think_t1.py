import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

div_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void div_leaky_relu_kernel(const float* input, float* output, int n, float inv_divisor, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] * inv_divisor;
        output[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor div_leaky_relu_cuda(torch::Tensor input, float divisor, float negative_slope) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    float inv_divisor = 1.0f / divisor;
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    div_leaky_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        inv_divisor,
        negative_slope
    );
    
    return output;
}
"""

div_leaky_relu_cpp = "torch::Tensor div_leaky_relu_cuda(torch::Tensor input, float divisor, float negative_slope);"

div_leaky_relu = load_inline(
    name="div_leaky_relu",
    cpp_sources=div_leaky_relu_cpp,
    cuda_sources=div_leaky_relu_source,
    functions=["div_leaky_relu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.negative_slope = 0.01

    def forward(self, x):
        x = self.conv(x)
        x = div_leaky_relu.div_leaky_relu_cuda(x, self.divisor, self.negative_slope)
        return x