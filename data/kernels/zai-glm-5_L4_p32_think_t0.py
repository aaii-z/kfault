import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scale and min reduction
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void scale_min_kernel(const float* __restrict__ input, float scale, float* __restrict__ output, int N, int C, int H, int W) {
    // Map block and thread indices to output coordinates (N, H, W)
    // We parallelize over the output dimensions. Each thread computes the min for one pixel (n, h, w).
    int n = blockIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (n < N && h < H && w < W) {
        // Calculate the base offset for the (n, h, w) location
        // Input layout is NCHW
        long long base_offset = (long long)n * C * H * W + (long long)h * W + w;
        long long channel_stride = (long long)H * W;

        float min_val = INFINITY;

        // Iterate over the channel dimension
        for (int c = 0; c < C; ++c) {
            float val = __ldg(&input[base_offset + c * channel_stride]);
            val *= scale;
            
            // Handle NaN propagation to match PyTorch behavior
            if (isnan(val)) {
                min_val = val;
                break; 
            }
            if (val < min_val) {
                min_val = val;
            }
        }
        
        // Output layout is N, 1, H, W
        output[(long long)n * H * W + h * W + w] = min_val;
    }
}

torch::Tensor scale_min_cuda(torch::Tensor input, float scale) {
    // Input dimensions
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Output tensor shape: (N, 1, H, W)
    auto output = torch::empty({N, 1, H, W}, input.options());

    // Grid and block dimensions
    // We use a 3D grid: x for N, y for H tiles, z for W tiles
    // Block size: 16x16 threads for spatial dimensions
    dim3 block(1, 16, 16);
    dim3 grid(N, (H + 15) / 16, (W + 15) / 16);

    scale_min_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        scale,
        output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}
"""

cpp_source = """
torch::Tensor scale_min_cuda(torch::Tensor input, float scale);
"""

# Compile the inline CUDA code
scale_min_op = load_inline(
    name="scale_min_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["scale_min_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, scales the output, and applies a minimum operation
    using a custom fused CUDA kernel for the scaling and min reduction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.scale_min_op = scale_min_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Standard convolution
        x = self.conv(x)
        
        # Fused scale and min operation
        # The custom kernel handles x * scale_factor and torch.min(x, dim=1, keepdim=True)[0]
        x = self.scale_min_op.scale_min_cuda(x, self.scale_factor)
        
        return x