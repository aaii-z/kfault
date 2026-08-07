import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Swish, Group Normalization, and HardSwish
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants for HardSwish
const float HARDSWISH_THRESHOLD = 3.0f;

// Kernel for fused Swish, Group Normalization, and HardSwish
__global__ void fused_swish_gn_hardswish_kernel(
    const float* __restrict__ x, 
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int D, int H, int W, 
    int groups, int channels_per_group, float eps) {
    
    // Calculate dimensions
    int spatial_size = D * H * W;
    int total_elements = N * C * spatial_size;
    
    // Each block handles one (n, g) pair for normalization
    int n = blockIdx.x / groups;
    int g = blockIdx.x % groups;
    
    if (n >= N) return;
    
    int c_start = g * channels_per_group;
    int c_end = c_start + channels_per_group;
    int group_size = channels_per_group * spatial_size;
    
    // Shared memory for reduction (mean and var)
    extern __shared__ float shared_data[];
    float* shared_sum = shared_data;
    float* shared_sum_sq = &shared_data[blockDim.x];
    
    // Step 1: Compute mean and variance for the group
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Grid-stride loop over the group's elements
    int group_start_idx = (n * C + c_start) * spatial_size;
    int group_end_idx = (n * C + c_end) * spatial_size;
    
    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        // Calculate the actual index in the tensor
        int c_local = idx / spatial_size;
        int spatial_idx = idx % spatial_size;
        int c = c_start + c_local;
        int tensor_idx = (n * C + c) * spatial_size + spatial_idx;
        
        float val = x[tensor_idx];
        
        // Apply Swish: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        float swish_val = val * sigmoid_val;
        
        local_sum += swish_val;
        local_sum_sq += swish_val * swish_val;
    }
    
    // Reduction in shared memory
    shared_sum[threadIdx.x] = local_sum;
    shared_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();
    
    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Compute mean and variance
    float mean = shared_sum[0] / group_size;
    float var = shared_sum_sq[0] / group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Step 2: Normalize, apply affine transform, and HardSwish
    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c_local = idx / spatial_size;
        int spatial_idx = idx % spatial_size;
        int c = c_start + c_local;
        int tensor_idx = (n * C + c) * spatial_size + spatial_idx;
        
        float val = x[tensor_idx];
        
        // Apply Swish
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        float swish_val = val * sigmoid_val;
        
        // Normalize
        float normalized = (swish_val - mean) * inv_std;
        
        // Apply affine transform (weight and bias)
        float gn_out = normalized * weight[c] + bias[c];
        
        // Apply HardSwish
        float hardswish_val;
        if (gn_out <= -HARDSWISH_THRESHOLD) {
            hardswish_val = 0.0f;
        } else if (gn_out >= HARDSWISH_THRESHOLD) {
            hardswish_val = gn_out;
        } else {
            hardswish_val = gn_out * (gn_out + HARDSWISH_THRESHOLD) / 6.0f;
        }
        
        out[tensor_idx] = hardswish_val;
    }
}

torch::Tensor fused_swish_gn_hardswish_cuda(
    torch::Tensor x, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int groups, float eps) {
    
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    int channels_per_group = C / groups;
    
    auto out = torch::empty_like(x);
    
    // Launch configuration
    int blocks = N * groups;
    int threads = 256;  // Power of 2 for reduction
    int shared_mem_size = 2 * threads * sizeof(float);
    
    fused_swish_gn_hardswish_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, D, H, W,
        groups, channels_per_group, eps
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_swish_gn_hardswish_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int groups, float eps);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_swish_gn_hardswish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by a fused
    Swish activation, group normalization, and HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused Swish + GroupNorm + HardSwish
        x = self.fused_ops.fused_swish_gn_hardswish_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.group_norm.num_groups, 
            self.group_norm.eps
        )
        return x