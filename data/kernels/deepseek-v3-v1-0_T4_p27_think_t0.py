import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel implementation
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

#define CUDA_NUM_THREADS 256
#define THREADS_PER_BLOCK 256

inline __device__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void group_norm_mean_reduce_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int C,
    int D,
    int H,
    int W,
    int G,
    float eps
) {
    int group_size = C / G;
    int spatial_size = D * H * W;
    int elements_per_group = group_size * spatial_size;
    
    extern __shared__ float shared[];
    float* group_means = shared;
    float* group_vars = &shared[G];
    
    int batch = blockIdx.x;
    int group = blockIdx.y;
    int tid = threadIdx.x;
    
    // Compute group mean and variance
    float mean = 0.0f;
    float var = 0.0f;
    
    int group_start = group * group_size;
    int group_end = group_start + group_size;
    
    // First pass: compute mean
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        int c = group_start + (i % group_size);
        int spatial_idx = i / group_size;
        int d = spatial_idx / (H * W);
        int h = (spatial_idx % (H * W)) / W;
        int w = spatial_idx % W;
        
        int idx = ((batch * C + c) * D + d) * H * W + h * W + w;
        float val = hardswish(input[idx]);
        mean += val;
    }
    
    mean = blockReduceSum(mean);
    if (tid == 0) {
        mean /= elements_per_group;
        group_means[group] = mean;
    }
    __syncthreads();
    
    // Second pass: compute variance
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        int c = group_start + (i % group_size);
        int spatial_idx = i / group_size;
        int d = spatial_idx / (H * W);
        int h = (spatial_idx % (H * W)) / W;
        int w = spatial_idx % W;
        
        int idx = ((batch * C + c) * D + d) * H * W + h * W + w;
        float val = hardswish(input[idx]);
        var += (val - mean) * (val - mean);
    }
    
    var = blockReduceSum(var);
    if (tid == 0) {
        var = var / elements_per_group + eps;
        group_vars[group] = var;
    }
    __syncthreads();
    
    // Normalize and reduce
    float inv_std = rsqrtf(group_vars[group]);
    float channel_sum[THREADS_PER_BLOCK / 32] = {0};
    
    for (int c = group_start; c < group_end; c++) {
        float channel_acc = 0.0f;
        
        for (int i = tid; i < spatial_size; i += blockDim.x) {
            int d = i / (H * W);
            int h = (i % (H * W)) / W;
            int w = i % W;
            
            int idx = ((batch * C + c) * D + d) * H * W + h * W + w;
            float val = hardswish(input[idx]);
            float normalized = (val - group_means[group]) * inv_std;
            normalized = normalized * weight[c] + bias[c];
            channel_acc += normalized;
        }
        
        // Reduce within warp
        for (int offset = 16; offset > 0; offset /= 2) {
            channel_acc += __shfl_down_sync(0xffffffff, channel_acc, offset);
        }
        
        if (tid % 32 == 0) {
            channel_sum[tid / 32] = channel_acc;
        }
        __syncthreads();
        
        // Reduce warps to single value
        if (tid < 32) {
            float warp_sum = 0.0f;
            for (int i = 0; i < blockDim.x / 32; i++) {
                warp_sum += channel_sum[i];
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            }
            if (tid == 0) {
                output[batch * C + c] = warp_sum / spatial_size;
            }
        }
        __syncthreads();
    }
}

__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

torch::Tensor group_norm_mean_reduce_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int G = num_groups;
    
    auto output = torch::zeros({B, C}, input.options());
    
    dim3 blocks(B, G);
    int threads = THREADS_PER_BLOCK;
    size_t shared_mem = 2 * G * sizeof(float);
    
    group_norm_mean_reduce_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W, G, eps
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor group_norm_mean_reduce_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps);"

# Load the custom CUDA extension
group_norm_mean_reduce = load_inline(
    name="group_norm_mean_reduce",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["group_norm_mean_reduce_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        # Use custom fused kernel for HardSwish + GroupNorm + Mean Reduction
        return group_norm_mean_reduce.group_norm_mean_reduce_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.num_groups, 
            self.eps
        )