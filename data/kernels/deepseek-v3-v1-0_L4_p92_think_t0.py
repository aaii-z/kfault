import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused GroupNorm-Tanh-HardSwish-Residual
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_NUM_THREADS 256

__device__ __forceinline__ float hard_swish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void group_norm_tanh_hardswish_residual_kernel(
    const float* input,
    const float* conv_output,
    float* output,
    const float* weight,
    const float* bias,
    int N,
    int C,
    int HxW,
    int group_size,
    int groups,
    float eps
) {
    int n = blockIdx.x;
    int group_id = blockIdx.y;
    int channel_offset = group_id * group_size;
    int actual_group_size = min(group_size, C - channel_offset);
    
    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = &shared[1];
    
    // Compute mean and variance for the group
    if (threadIdx.x == 0) {
        mean_shared[0] = 0.0f;
        var_shared[0] = 0.0f;
    }
    __syncthreads();
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int num_elems = actual_group_size * HxW;
    
    for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
        int c = channel_offset + (i / HxW) % actual_group_size;
        int hw = i % HxW;
        int idx = n * C * HxW + c * HxW + hw;
        float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }
    
    atomicAdd(mean_shared, sum);
    atomicAdd(var_shared, sum_sq);
    __syncthreads();
    
    float mean = mean_shared[0] / num_elems;
    float var = var_shared[0] / num_elems - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Apply GroupNorm, Tanh, HardSwish, and Residual
    for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
        int c = channel_offset + (i / HxW) % actual_group_size;
        int hw = i % HxW;
        int idx = n * C * HxW + c * HxW + hw;
        
        float norm_val = (input[idx] - mean) * inv_std;
        norm_val = norm_val * weight[c] + bias[c];
        float tanh_val = tanhf(norm_val);
        float hs_val = hard_swish(tanh_val);
        output[idx] = conv_output[idx] + hs_val;
    }
}

torch::Tensor fused_gnthr_cuda(
    torch::Tensor input,
    torch::Tensor conv_output,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float eps
) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int HxW = H * W;
    int group_size = C / groups;
    
    auto output = torch::empty_like(input);
    
    dim3 grid(N, groups);
    int shared_mem = 2 * sizeof(float);
    
    group_norm_tanh_hardswish_residual_kernel<<<grid, CUDA_NUM_THREADS, shared_mem>>>(
        input.data_ptr<float>(),
        conv_output.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, HxW, group_size, groups, eps
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor fused_gnthr_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, float);"

fused_gnthr = load_inline(
    name='fused_gnthr',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_gnthr_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.fused_gnthr = fused_gnthr

    def forward(self, x):
        x_conv = self.conv(x)
        x_fused = self.fused_gnthr.fused_gnthr_cuda(
            x_conv,
            x_conv,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps
        )
        return torch.logsumexp(x_fused, dim=1, keepdim=True)