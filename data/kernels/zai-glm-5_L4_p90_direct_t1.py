import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LeakyReLU, Add, Clamp, and GELU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Approximation for GELU (Gaussian Error Linear Unit)
__device__ __forceinline__ float gelu_approx(float x) {
    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x)));
}

__global__ void fused_ops_kernel(const float* __restrict__ x, const float* __restrict__ sum_tensor, float* __restrict__ out, int N, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * D * H * W;
    
    if (idx < total_size) {
        // Decompose linear index into (n, c, d, h, w)
        // x is contiguous (N, C, D, H, W)
        int w = idx % W;
        int h = (idx / W) % H;
        int d = (idx / (W * H)) % D;
        int c = (idx / (W * H * D)) % C;
        int n = idx / (W * H * D * C);

        float val = x[idx];
        
        // LeakyReLU with negative_slope=0.2
        val = (val >= 0.0f) ? val : 0.2f * val;
        
        // Add sum_tensor. sum_tensor has shape (out_channels, 1, 1, 1)
        // We access the value corresponding to channel c.
        // sum_tensor is stored as (C, 1, 1, 1), so linear index is c.
        val += sum_tensor[c];
        
        // Clamp between -1.0 and 1.0
        val = fmaxf(-1.0f, fminf(1.0f, val));
        
        // GELU activation
        out[idx] = gelu_approx(val);
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor sum_tensor) {
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    auto out = torch::empty_like(x);
    int total_size = x.numel();

    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        sum_tensor.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, C, D, H, W
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor sum_tensor);"
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
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    The post-convolution operations are fused into a single custom CUDA kernel for performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # The custom kernel handles broadcasting of sum_tensor (C, 1, 1, 1) 
        # against the conv output (N, C, D, H, W).
        x = self.fused_ops.fused_ops_cuda(x, self.sum_tensor)
        return x