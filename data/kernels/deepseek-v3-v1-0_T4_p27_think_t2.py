import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define THREADS_PER_BLOCK 256

inline __device__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
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

__global__ void hardswish_group_norm_kernel(
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
    
    int batch = blockIdx.x;
    int group = blockIdx.y;
    int tid = threadIdx.x;
    
    int group_start = group * group_size;
    
    // Compute group mean
    float mean = 0.0f;
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
    }
    __syncthreads();
    
    // Compute group variance
    float var = 0.0f;
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
    }
    __syncthreads();
    
    // Normalize and write output
    float inv_std = rsqrtf(var);
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        int c = group_start + (i % group_size);
        int spatial_idx = i / group_size;
        int d = spatial_idx / (H * W);
        int h = (spatial_idx % (H * W)) / W;
        int w = spatial_idx % W;
        
        int idx = ((batch * C + c) * D + d) * H * W + h * W + w;
        float val = hardswish(input[idx]);
        float normalized = (val - mean) * inv_std;
        normalized = normalized * weight[c] + bias[c];
        output[idx] = normalized;
    }
}

__global__ void mean_reduction_kernel(
    const float* input,
    float* output,
    int B,
    int C,
    int D,
    int H,
    int W
) {
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int spatial_size = D * H * W;
    
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int d = i / (H * W);
        int h = (i % (H * W)) / W;
        int w = i % W;
        int idx = ((batch * C + channel) * D + d) * H * W + h * W + w;
        sum += input[idx];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch * C + channel] = sdata[0] / spatial_size;
    }
}

torch::Tensor hardswish_group_norm_cuda(
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
    
    auto output = torch::empty_like(input);
    
    dim3 blocks(B, G);
    int threads = THREADS_PER_BLOCK;
    
    hardswish_group_norm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W, G, eps
    );
    
    return output;
}

torch::Tensor mean_reduction_cuda(torch::Tensor input) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    auto output = torch::zeros({B, C}, input.options());
    
    dim3 blocks(B, C);
    int threads = THREADS_PER_BLOCK;
    int shared_mem = threads * sizeof(float);
    
    mean_reduction_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor hardswish_group_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps);
torch::Tensor mean_reduction_cuda(torch::Tensor input);
"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["hardswish_group_norm_cuda", "mean_reduction_cuda"],
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
        x = custom_ops.hardswish_group_norm_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.num_groups, 
            self.eps
        )
        x = custom_ops.mean_reduction_cuda(x)
        return x