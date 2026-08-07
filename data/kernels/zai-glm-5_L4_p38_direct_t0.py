import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for fused clamp, softmax, and scale
__global__ void fused_ops_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ scale,
    float clamp_min,
    float clamp_max,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width) {
    
    int spatial_size = depth * height * width;
    int total_elements = batch_size * channels * spatial_size;
    
    // Each block handles one (batch, channel) pair
    int bc_idx = blockIdx.x;
    if (bc_idx >= batch_size * channels) return;

    int b = bc_idx / channels;
    int c = bc_idx % channels;
    
    int base_offset = (b * channels + c) * spatial_size;
    
    // Step 1: Clamp and find max (for numerical stability)
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = x[base_offset + i];
        val = fmaxf(clamp_min, fminf(clamp_max, val));
        // We need to store the clamped value temporarily or re-compute.
        // Re-computing is cheaper than shared memory write/read for this size usually.
        // But we need the clamped value for the exp sum later.
        // Let's use shared memory to store clamped values if needed, or just recompute.
        // Given the potential size (32*64*64 = 131072), shared memory might be tight if we store all.
        // However, typical spatial dims might be smaller or larger.
        // Let's just recompute clamp in the second pass.
        
        // AtomicMax is slow. We can use shared memory reduction for max.
        // But for simplicity and given the problem constraints, let's do a block-stride loop for max.
        // Actually, we can do a parallel reduction in shared memory.
        
        // Simple approach: each thread finds max in its stride, then reduce in shared mem.
        if (val > max_val) {
            max_val = val;
        }
    }
    
    // Shared memory reduction for max
    extern __shared__ float s_data[];
    s_data[threadIdx.x] = max_val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] = fmaxf(s_data[threadIdx.x], s_data[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float block_max = s_data[0];
    __syncthreads();

    // Step 2: Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = x[base_offset + i];
        val = fmaxf(clamp_min, fminf(clamp_max, val));
        sum_exp += expf(val - block_max);
    }
    
    s_data[threadIdx.x] = sum_exp;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    float block_sum_exp = s_data[0];
    
    // Step 3: Compute softmax and apply scale
    float scale_val = scale[c]; // Scale is (1, C, 1, 1, 1)
    
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = x[base_offset + i];
        val = fmaxf(clamp_min, fminf(clamp_max, val));
        float softmax_val = expf(val - block_max) / block_sum_exp;
        out[base_offset + i] = softmax_val * scale_val;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    double clamp_min,
    double clamp_max) {
    
    auto sizes = x.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    
    auto out = torch::empty_like(x);
    
    int total_spatial = depth * height * width;
    
    // Launch configuration
    // One block per (batch, channel)
    int blocks = batch_size * channels;
    int threads = 256; // Safe bet for block size
    
    // Shared memory size for reduction
    size_t shared_mem_size = threads * sizeof(float);
    
    fused_ops_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        scale.data_ptr<float>(),
        static_cast<float>(clamp_min),
        static_cast<float>(clamp_max),
        batch_size,
        channels,
        depth,
        height,
        width
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor scale, double clamp_min, double clamp_max);"
)

# Compile the inline CUDA code
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
    Optimized Model that performs average pooling, 3D transposed convolution,
    and a fused kernel for clamping, softmax, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        
        # Apply fused kernel
        x = self.fused_ops.fused_ops_cuda(x, self.scale, self.clamp_min, self.clamp_max)
        
        return x