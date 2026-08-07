import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused activation kernel (Mish + Tanh)
fused_activation_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__device__ float mish_tanh_activation(float x) {
    // Mish activation: x * tanh(softplus(x))
    float sp = log1pf(expf(x));  // softplus(x) = log(1 + exp(x))
    float mish = x * tanhf(sp);
    return tanhf(mish);  // Apply tanh to Mish output
}

__global__ void fused_activation_kernel(const float* input, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = mish_tanh_activation(input[idx]);
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    const int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    fused_activation_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        x = self.conv(x)
        x = fused_activation_ext.fused_activation_cuda(x)
        return x