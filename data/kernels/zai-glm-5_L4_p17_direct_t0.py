import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Instance Normalization and Division
# This kernel fuses the normalization calculation (mean, var, normalize) and the final division
# to maximize memory bandwidth utilization.
fused_instancenorm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for Instance Normalization + Division
// Optimized for FP32.
// Uses shared memory to reduce global memory traffic for mean/var calculation.
__global__ void instance_norm_div_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    int N, int C, int H, int W, 
    float divide_val, 
    float epsilon) 
{
    // Each block handles one (n, c) instance
    // Block size is assumed to be a multiple of 32 and <= 1024
    int n = blockIdx.x / C;
    int c = blockIdx.x % C;
    
    int instance_size = H * W;
    int tid = threadIdx.x;
    
    // Pointers to the start of the current instance
    const float* x_ptr = input + (n * C + c) * instance_size;
    float* y_ptr = output + (n * C + c) * instance_size;
    
    // Shared memory for partial sums
    // Size depends on block size, max 1024 floats
    extern __shared__ float s_mem[];
    float* s_sum = s_mem;
    float* s_sum_sq = s_mem + blockDim.x;
    
    // Step 1: Calculate Mean and Variance
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Grid-stride loop to handle arbitrary H*W sizes
    for (int i = tid; i < instance_size; i += blockDim.x) {
        float val = x_ptr[i];
        sum += val;
        sum_sq += val * val;
    }
    
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();
    
    // Parallel reduction in shared memory
    // Restrict to the first warp or power-of-2 threads depending on size
    // Simple tree reduction for clarity and performance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 computes final mean and inv_std
    float mean = 0.0f;
    float inv_std = 0.0f;
    
    if (tid == 0) {
        mean = s_sum[0] / instance_size;
        float var = (s_sum_sq[0] / instance_size) - (mean * mean);
        // Clamp variance to avoid sqrt(0) or negative due to precision
        var = fmaxf(var, 0.0f);
        inv_std = rsqrtf(var + epsilon);
    }
    // Broadcast mean and inv_std to all threads in the block
    // Using volatile float pointer to ensure visibility without __syncthreads if we were in a warp
    // But here we need it for the whole block for the next phase
    __syncthreads(); 
    
    // We can use shared memory to broadcast or just re-compute if we had enough registers, 
    // but shared memory is safer for broadcasting from thread 0.
    // However, since we computed in thread 0, we can write to a specific shared location
    // Actually, we can just use the s_mem[0] and s_mem[1] which hold sum and sum_sq? No, those are partials.
    // Let's write mean and inv_std to specific shared addresses.
    __shared__ float s_mean, s_inv_std;
    if (tid == 0) {
        s_mean = mean;
        s_inv_std = inv_std;
    }
    __syncthreads();
    
    mean = s_mean;
    inv_std = s_inv_std;
    
    // Step 2: Normalize and Divide
    // We fuse the division here: y = (x - mean) * inv_std / divide_val
    float combined_scale = inv_std / divide_val;
    
    for (int i = tid; i < instance_size; i += blockDim.x) {
        float val = x_ptr[i];
        y_ptr[i] = (val - mean) * combined_scale;
    }
}

torch::Tensor fused_instancenorm_div_cuda(
    torch::Tensor input, 
    float divide_val, 
    float epsilon) 
{
    auto output = torch::empty_like(input);
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    // Launch configuration
    // One block per (N, C) instance
    int blocks = N * C;
    
    // Block size heuristic:
    // We need enough threads to cover H*W efficiently.
    // 128 or 256 threads is usually a sweet spot for reduction + memory access.
    // We cap at 1024 threads per block.
    int block_size = 256; 
    if (H * W <= 1024) {
        // Find next power of 2 >= H*W, or just use H*W if small?
        // Using a fixed reasonable size like 256 handles small images well enough via grid-stride loop.
        // If H*W is very small (e.g. 1x1), 256 is overkill but safe.
        // If H*W is large, 256 provides good occupancy.
        block_size = 256;
    } else {
        block_size = 512; // For larger images
    }
    
    // Shared memory size: 2 * block_size * sizeof(float)
    size_t shared_mem_size = 2 * block_size * sizeof(float);
    
    instance_norm_div_kernel<<<blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        divide_val,
        epsilon
    );
    
    return output;
}
"""

fused_instancenorm_div_cpp_source = (
    "torch::Tensor fused_instancenorm_div_cuda(torch::Tensor input, float divide_val, float epsilon);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_instancenorm_div",
    cpp_sources=fused_instancenorm_div_cpp_source,
    cuda_sources=fused_instancenorm_div_source,
    functions=["fused_instancenorm_div_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution followed by a fused 
    Instance Normalization and Division custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # We remove the standard nn.InstanceNorm2d
        self.divide_by = divide_by
        self.fused_ops = fused_ops
        self.epsilon = 1e-5 # Standard epsilon for InstanceNorm

    def forward(self, x):
        x = self.conv(x)
        # Apply the fused kernel
        x = self.fused_ops.fused_instancenorm_div_cuda(x, self.divide_by, self.epsilon)
        return x