import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm, Scale, MaxPool, and Clamp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void compute_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ stats,
    int N, int C, int H, int W,
    int G, int C_per_G) {

    int n = blockIdx.x;
    int g = blockIdx.y;
    int c_start = g * C_per_G;
    
    // Shared memory for reduction
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    
    int tid = threadIdx.x;
    s_sum[tid] = 0.0f;
    s_sum_sq[tid] = 0.0f;
    
    int total_pixels = C_per_G * H * W;
    
    // Grid stride loop for reduction
    // Each thread processes a strided portion of the data
    long long base_offset = (long long)n * C * H * W + (long long)c_start * H * W;
    
    for (int i = tid; i < total_pixels; i += blockDim.x) {
        float val = x[base_offset + i];
        s_sum[tid] += val;
        s_sum_sq[tid] += val * val;
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float mean = s_sum[0] / total_pixels;
        float var = s_sum_sq[0] / total_pixels - mean * mean;
        stats[(n * G + g) * 2 + 0] = mean;
        stats[(n * G + g) * 2 + 1] = var;
    }
}

__global__ void fused_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ stats,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int G, int C_per_G,
    int H_out, int W_out,
    int PoolSize,
    float clamp_min, float clamp_max) {

    int h_out = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;
    
    if (h_out >= H_out || w_out >= W_out) return;
    
    int n = nc / C;
    int c = nc % C;
    
    int g = c / C_per_G;
    
    // Load stats
    float mean = stats[(n * G + g) * 2 + 0];
    float var = stats[(n * G + g) * 2 + 1];
    
    // Load params
    float g_val = gamma[c];
    float b_val = beta[c];
    float s_val = scale[c];
    
    float inv_std = rsqrtf(var + 1e-5f);
    
    float max_val = -INFINITY;
    
    int h_in_start = h_out * PoolSize;
    int w_in_start = w_out * PoolSize;
    
    // Loop over pooling window
    for (int kh = 0; kh < PoolSize; ++kh) {
        int h_in = h_in_start + kh;
        if (h_in >= H) continue;
        
        for (int kw = 0; kw < PoolSize; ++kw) {
            int w_in = w_in_start + kw;
            if (w_in >= W) continue;
            
            long long idx = (long long)n * C * H * W + c * H * W + h_in * W + w_in;
            float val = x[idx];
            
            // GroupNorm
            float norm_val = (val - mean) * inv_std;
            
            // Affine
            float affine_val = norm_val * g_val + b_val;
            
            // Scale
            float scaled_val = affine_val * s_val;
            
            // MaxPool
            max_val = fmaxf(max_val, scaled_val);
        }
    }
    
    // Clamp
    float final_val = fmaxf(clamp_min, fminf(clamp_max, max_val));
    
    long long out_idx = (long long)n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out;
    out[out_idx] = final_val;
}

torch::Tensor fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor scale,
    int num_groups,
    int pool_size,
    float clamp_min,
    float clamp_max) {

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_per_G = C / num_groups;
    
    // 1. Compute Stats (Mean, Var)
    // Output: stats tensor of shape (N, G, 2)
    auto stats_options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto stats = torch::empty({N, num_groups, 2}, stats_options);
    
    const int block_size_stats = 256;
    dim3 grid_stats(N, num_groups);
    
    compute_stats_kernel<<<grid_stats, block_size_stats>>>(
        x.data_ptr<float>(),
        stats.data_ptr<float>(),
        N, C, H, W,
        num_groups, C_per_G
    );
    
    // 2. Fused Forward
    int H_out = (H - pool_size) / pool_size + 1;
    int W_out = (W - pool_size) / pool_size + 1;
    
    auto out = torch::empty({N, C, H_out, W_out}, x.options());
    
    const int block_size_fwd = 16;
    dim3 grid_fwd((H_out + block_size_fwd - 1) / block_size_fwd, 
                  (W_out + block_size_fwd - 1) / block_size_fwd, 
                  N * C);
    
    fused_forward_kernel<<<grid_fwd, dim3(block_size_fwd, block_size_fwd)>>>(
        x.data_ptr<float>(),
        stats.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        num_groups, C_per_G,
        H_out, W_out,
        pool_size,
        clamp_min, clamp_max
    );
    
    return out;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor scale,
    int num_groups,
    int pool_size,
    float clamp_min,
    float clamp_max);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs convolution, followed by a fused kernel
    for GroupNorm, Scaling, Max Pooling, and Clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        
        # Use the fused CUDA kernel for the remaining operations
        # We extract weight and bias from group_norm to pass to the kernel
        x = self.fused_ops.fused_forward_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.scale, 
            self.group_norm.num_groups,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max
        )
        return x