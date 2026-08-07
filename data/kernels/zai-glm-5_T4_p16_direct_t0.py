import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
__device__ __forceinline__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

// Hardtanh activation: clamp(x, min_val, max_val)
__device__ __forceinline__ float hardtanh_activation(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__global__ void fused_ops_kernel(const float* __restrict__ x, float* __restrict__ out, float add_value, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Apply Mish
        val = mish_activation(val);
        // Add value
        val += add_value;
        // Apply Hardtanh (min=-1, max=1)
        val = hardtanh_activation(val, -1.0f, 1.0f);
        // Scale
        val *= scale;
        out[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, float add_value, float scale) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), add_value, scale, size);

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, float add_value, float scale);"
)

# Compile the inline CUDA code for fused operations
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution followed by a fused kernel
    for Mish, addition, Hardtanh, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused Mish + Add + Hardtanh + Scale
        x = self.fused_ops.fused_ops_cuda(x, self.add_value, self.scale)
        return x