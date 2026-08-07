import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused activation: softplus + tanh + multiply
fused_activation_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_activation_kernel(
    const float* input,
    float* output,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float x = input[idx];
        float sp = logf(1.0f + expf(x));
        float tanh_sp = tanhf(sp);
        output[idx] = tanh_sp * x;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_activation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements
    );
    
    return output;
}
"""

fused_activation_cpp = "torch::Tensor fused_activation_cuda(torch::Tensor input);"

# Load the custom CUDA extension
fused_activation_ext = load_inline(
    name='fused_activation',
    cpp_sources=fused_activation_cpp,
    cuda_sources=fused_activation_source,
    functions=['fused_activation_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = fused_activation_ext.fused_activation_cuda(x)
        x = self.bn(x)
        return x