import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused division and LeakyReLU
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_div_leaky_relu_kernel(const float* __restrict__ input, float* __restrict__ output, int size, float inv_divisor, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        float divided = val * inv_divisor;
        // LeakyReLU: x if x > 0 else x * negative_slope
        output[idx] = divided > 0.0f ? divided : divided * negative_slope;
    }
}

torch::Tensor fused_div_leaky_relu_cuda(torch::Tensor input, float divisor, float negative_slope) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // Precompute inverse for faster multiplication
    float inv_divisor = 1.0f / divisor;

    fused_div_leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, 
        inv_divisor, 
        negative_slope
    );
    
    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_div_leaky_relu_cuda(torch::Tensor input, float divisor, float negative_slope);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_div_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, followed by a fused division and LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        # Apply fused division and LeakyReLU
        # negative_slope is fixed at 0.01 as per the original model
        x = self.fused_op.fused_div_leaky_relu_cuda(x, self.divisor, 0.01)
        return x