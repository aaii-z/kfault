import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: Mish + Add + Hardtanh + Scale
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish_activation(float x) {
    // Mish activation: x * tanh(softplus(x))
    // softplus(x) = ln(1 + exp(x))
    // Using log1pf(expf(x)) for numerical stability
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_mish_add_hardtanh_scale_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int size, 
    float add_value, 
    float scale) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        
        // Apply Mish
        val = mish_activation(val);
        
        // Add value
        val += add_value;
        
        // Apply Hardtanh (clamp between -1 and 1)
        val = fmaxf(-1.0f, fminf(1.0f, val));
        
        // Scale
        val *= scale;
        
        output[idx] = val;
    }
}

torch::Tensor fused_mish_add_hardtanh_scale_cuda(torch::Tensor input, float add_value, float scale) {
    auto size = input.numel();
    auto out = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_mish_add_hardtanh_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size, 
        add_value, 
        scale
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_mish_add_hardtanh_scale_cuda(torch::Tensor input, float add_value, float scale);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_mish_add_hardtanh_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        # Ensure input is contiguous for the CUDA kernel
        x = self.fused_ops.fused_mish_add_hardtanh_scale_cuda(x.contiguous(), self.add_value, self.scale)
        return x