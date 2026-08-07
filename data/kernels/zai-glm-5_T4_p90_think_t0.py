import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LeakyReLU, Add, Clamp, and GELU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_postprocess_kernel(const float* __restrict__ input, const float* __restrict__ sum_tensor, float* __restrict__ output, int size, int C, int DHW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        
        // LeakyReLU (negative_slope=0.2)
        if (val < 0.0f) {
            val *= 0.2f;
        }
        
        // Add sum_tensor
        // sum_tensor shape is (C, 1, 1, 1), broadcasting across (N, C, D, H, W)
        // We need to determine the channel index 'c' from the linear index 'idx'
        // idx corresponds to (n, c, d, h, w) flattened
        // The layout is contiguous, so:
        // channel_idx = (idx / DHW) % C
        int channel_idx = (idx / DHW) % C;
        val += sum_tensor[channel_idx];
        
        // Clamp (-1.0, 1.0)
        val = fmaxf(-1.0f, fminf(1.0f, val));
        
        // GELU (exact approximation using erf)
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        val = 0.5f * val * (1.0f + erff(val * 0.70710678118654752440f));
        
        output[idx] = val;
    }
}

torch::Tensor fused_postprocess_cuda(torch::Tensor input, torch::Tensor sum_tensor) {
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    int DHW = D * H * W;
    int size = input.numel();
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_postprocess_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        sum_tensor.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, 
        C, 
        DHW
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_postprocess_cuda(torch::Tensor input, torch::Tensor sum_tensor);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_postprocess_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution followed by a fused custom CUDA kernel
    for LeakyReLU, Add, Clamp, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # Apply fused kernel: LeakyReLU -> Add -> Clamp -> GELU
        x = self.fused_ops.fused_postprocess_cuda(x, self.sum_tensor)
        return x