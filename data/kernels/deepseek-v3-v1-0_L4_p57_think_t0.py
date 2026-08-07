import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused activation kernel (ReLU + HardSwish)
fused_activation_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_activation_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Fused ReLU + HardSwish
        if (x <= 0.0f) {
            output[idx] = 0.0f;
        } else if (x >= 3.0f) {
            output[idx] = x;
        } else {
            output[idx] = x * (x + 3.0f) / 6.0f;
        }
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    fused_activation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}
"""

fused_activation_cpp_source = "torch::Tensor fused_activation_cuda(torch::Tensor input);"

# Load the custom CUDA extension
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_activation_cuda(x)
        return x