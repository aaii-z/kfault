import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm and HardSwish
group_norm_hardswish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_NUM_THREADS 256

__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void group_norm_hardswish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int C,
    int D,
    int H,
    int W,
    int G,
    float eps
) {
    int n = blockIdx.x;
    int g = blockIdx.y;
    int spatial_size = D * H * W;
    int channels_per_group = C / G;
    
    extern __shared__ float sdata[];
    float* mean_shared = sdata;
    float* var_shared = &sdata[1];
    
    if (threadIdx.x == 0) {
        mean_shared[0] = 0.0f;
        var_shared[0] = 0.0f;
    }
    __syncthreads();
    
    // Compute mean
    float mean = 0.0f;
    for (int c = g * channels_per_group; c < (g + 1) * channels_per_group; c++) {
        for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
            int d = s / (H * W);
            int h = (s % (H * W)) / W;
            int w = s % W;
            int idx = n * C * spatial_size + c * spatial_size + d * H * W + h * W + w;
            mean += input[idx];
        }
    }
    mean /= (channels_per_group * spatial_size);
    
    atomicAdd(mean_shared, mean);
    __syncthreads();
    mean = mean_shared[0];
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mean_shared[0] = 0.0f;
    }
    __syncthreads();
    
    // Compute variance
    float var = 0.0f;
    for (int c = g * channels_per_group; c < (g + 1) * channels_per_group; c++) {
        for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
            int d = s / (H * W);
            int h = (s % (H * W)) / W;
            int w = s % W;
            int idx = n * C * spatial_size + c * spatial_size + d * H * W + h * W + w;
            float diff = input[idx] - mean;
            var += diff * diff;
        }
    }
    var /= (channels_per_group * spatial_size);
    
    atomicAdd(var_shared, var);
    __syncthreads();
    var = var_shared[0];
    
    // Normalize and apply HardSwish
    for (int c = g * channels_per_group; c < (g + 1) * channels_per_group; c++) {
        for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
            int d = s / (H * W);
            int h = (s % (H * W)) / W;
            int w = s % W;
            int idx = n * C * spatial_size + c * spatial_size + d * H * W + h * W + w;
            float normalized = (input[idx] - mean) / sqrtf(var + eps);
            float scaled = normalized * weight[c] + bias[c];
            output[idx] = hardswish(scaled);
        }
    }
}

torch::Tensor group_norm_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float eps
) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto output = torch::empty_like(input);
    
    dim3 blocks(N, groups);
    int threads = min(C * D * H * W, CUDA_NUM_THREADS);
    int shared_memory = 2 * sizeof(float);
    
    group_norm_hardswish_kernel<<<blocks, threads, shared_memory>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W, groups, eps
    );
    
    return output;
}
"""

group_norm_hardswish_cpp_source = """
torch::Tensor group_norm_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float eps
);
"""

# Compile the inline CUDA code
group_norm_hardswish = load_inline(
    name="group_norm_hardswish",
    cpp_sources=group_norm_hardswish_cpp_source,
    cuda_sources=group_norm_hardswish_source,
    functions=["group_norm_hardswish_cuda"],
    verbose=True
)

class GroupNormHardSwish(nn.Module):
    def __init__(self, num_channels, num_groups, eps):
        super(GroupNormHardSwish, self).__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        
    def forward(self, x):
        return group_norm_hardswish.group_norm_hardswish_cuda(
            x, self.weight, self.bias, self.num_groups, self.eps
        )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.group_norm_hardswish = GroupNormHardSwish(out_channels, groups, eps)
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = self.group_norm_hardswish(x)
        return x