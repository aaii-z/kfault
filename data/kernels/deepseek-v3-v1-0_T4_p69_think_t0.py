import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused HardSwish + ReLU CUDA kernel
fused_activation_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_hardswish_relu_kernel(const float* input, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x = input[idx];
        // HardSwish: x * relu6(x + 3) / 6
        float hardswish = x * min(max(0.0f, x + 3.0f), 6.0f) / 6.0f;
        // ReLU
        output[idx] = fmaxf(0.0f, hardswish);
    }
}

torch::Tensor fused_hardswish_relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    const int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    fused_hardswish_relu_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
    );
    
    return output;
}
"""

fused_activation_cpp_source = "torch::Tensor fused_hardswish_relu_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_hardswish_relu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_hardswish_relu_cuda(x)
        return x