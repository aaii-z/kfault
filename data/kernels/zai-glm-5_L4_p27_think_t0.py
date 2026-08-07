import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused HardSwish, GroupNorm, and Mean Pooling
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Compute Mean and Variance for GroupNorm
// Grid: (Batch, Groups)
// Block: 256 threads
__global__ void stats_kernel(const float* __restrict__ input,
                             float* __restrict__ mean_out,
                             float* __restrict__ var_out,
                             int B, int C, int G, int S) {
    // S: Spatial size (D*H*W)
    // C: Channels
    // G: Groups
    // CG: Channels per group
    int b = blockIdx.x;
    int g = blockIdx.y;
    
    int CG = C / G;
    int volume = CG * S;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Stride loop over the volume assigned to this group (channels + spatial)
    // The input is (B, C, D*H*W)
    // This block handles channels [g*CG, (g+1)*CG) for batch b
    for (int i = threadIdx.x; i < volume; i += blockDim.x) {
        int c_local = i / S;
        int s_idx = i % S;
        int c_global = g * CG + c_local;
        
        // Input index: (b, c_global, s_idx)
        // Assuming contiguous memory layout: b * C * S + c_global * S + s_idx
        float val = input[b * C * S + c_global * S + s_idx];
        
        // HardSwish: x * relu6(x + 3) / 6
        // relu6(x) = min(max(x, 0), 6)
        float relu6 = fminf(fmaxf(val + 3.0f, 0.0f), 6.0f);
        float hardswish_val = val * relu6 / 6.0f;
        
        sum += hardswish_val;
        sum_sq += hardswish_val * hardswish_val;
    }
    
    // Block reduction
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    
    s_sum[threadIdx.x] = sum;
    s_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float mean = s_sum[0] / volume;
        float var = s_sum_sq[0] / volume - mean * mean;
        mean_out[b * G + g] = mean;
        var_out[b * G + g] = var;
    }
}

// Kernel 2: Apply Normalization and Mean Pooling
// Grid: (Batch, Channels)
// Block: 256 threads
__global__ void norm_pool_kernel(const float* __restrict__ input,
                                 const float* __restrict__ mean_in,
                                 const float* __restrict__ var_in,
                                 const float* __restrict__ weight,
                                 const float* __restrict__ bias,
                                 float* __restrict__ out,
                                 int B, int C, int G, int S, float eps) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    
    int g = c / (C / G);
    
    // Load stats for this group
    float mean = mean_in[b * G + g];
    float var = var_in[b * G + g];
    float inv_std = rsqrtf(var + eps);
    
    // Load weight and bias for this channel
    float w = weight[c];
    float b_val = bias[c];
    
    float sum_pool = 0.0f;
    
    // Stride loop over spatial dimensions
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        float val = input[b * C * S + c * S + s];
        
        // HardSwish
        float relu6 = fminf(fmaxf(val + 3.0f, 0.0f), 6.0f);
        float hardswish_val = val * relu6 / 6.0f;
        
        // GroupNorm
        float norm_val = (hardswish_val - mean) * inv_std * w + b_val;
        
        sum_pool += norm_val;
    }
    
    // Block reduction for pooling
    __shared__ float s_pool[256];
    s_pool[threadIdx.x] = sum_pool;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_pool[threadIdx.x] += s_pool[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        out[b * C + c] = s_pool[0] / S;
    }
}

// C++ wrapper function
torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int num_groups, double eps) {
    // input: (B, C, D, H, W)
    // weight, bias: (C)
    
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int S = D * H * W;
    int G = num_groups;
    
    // Temporary storage for stats
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto mean_out = torch::empty({B, G}, options);
    auto var_out = torch::empty({B, G}, options);
    
    // Output tensor
    auto out = torch::empty({B, C}, options);
    
    const int block_size = 256;
    
    // Launch stats kernel
    // Grid: B x G
    dim3 grid_stats(B, G);
    stats_kernel<<<grid_stats, block_size>>>(
        input.data_ptr<float>(),
        mean_out.data_ptr<float>(),
        var_out.data_ptr<float>(),
        B, C, G, S
    );
    
    // Launch norm and pool kernel
    // Grid: B x C
    dim3 grid_norm(B, C);
    norm_pool_kernel<<<grid_norm, block_size>>>(
        input.data_ptr<float>(),
        mean_out.data_ptr<float>(),
        var_out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, G, S, static_cast<float>(eps)
    );
    
    return out;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int num_groups, double eps);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_op = fused_op

    def forward(self, x):
        # 1. Conv3D
        x = self.conv(x)
        
        # 2. Fused HardSwish + GroupNorm + Mean Pooling
        # We use the custom CUDA kernel here
        # Note: We pass the group_norm parameters (weight, bias) to the kernel
        x = self.fused_op.fused_forward_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.group_norm.num_groups, 
            self.group_norm.eps
        )
        return x