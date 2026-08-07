import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Mish and Tanh activation
# Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
# We fuse Mish and the final Tanh into a single kernel.
# The operation becomes: out = tanh(Mish(x))
fused_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_mish_tanh_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        
        // Compute Mish(val)
        // softplus(val) = log(1 + exp(val))
        // Using log1p(exp(val)) for numerical stability, though for large val, exp(val) might overflow.
        // However, tanh(softplus) handles large values gracefully (tanh approaches 1).
        // For val > 20, exp(val) is huge, softplus approx val, tanh(softplus) approx 1.
        // Mish(val) approx val * 1 = val.
        
        float sp;
        if (val > 20.0f) {
            sp = val;
        } else {
            sp = log1pf(expf(val));
        }
        
        float mish_val = val * tanhf(sp);
        
        // Compute Tanh(Mish(val))
        out[idx] = tanhf(mish_val);
    }
}

torch::Tensor fused_mish_tanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_mish_tanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size
    );

    return out;
}
"""

fused_mish_tanh_cpp_source = (
    "torch::Tensor fused_mish_tanh_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
fused_mish_tanh_op = load_inline(
    name="fused_mish_tanh",
    cpp_sources=fused_mish_tanh_cpp_source,
    cuda_sources=fused_mish_tanh_source,
    functions=["fused_mish_tanh_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution followed by a fused Mish and Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_op = fused_mish_tanh_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv(x)
        x = self.fused_op.fused_mish_tanh_cuda(x)
        return x