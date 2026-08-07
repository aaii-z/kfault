import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

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
    int channels_per_group = C / G;
    int group_size = channels_per_group * D * H * W;
    
    extern __shared__ float sdata[];
    float* shared = sdata;

    // Compute mean
    float mean = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c = g * channels_per_group + i / (D * H * W);
        int spatial = i % (D * H * W);
        int d = spatial / (H * W);
        int h = (spatial % (H * W)) / W;
        int w = spatial % W;
        int idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        mean += input[idx];
    }
    shared[threadIdx.x] = mean;
    __syncthreads();

    // Reduce mean
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    mean = shared[0] / group_size;
    __syncthreads();

    // Compute variance
    float variance = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c = g * channels_per_group + i / (D * H * W);
        int spatial = i % (D * H * W);
        int d = spatial / (H * W);
        int h = (spatial % (H * W)) / W;
        int w = spatial % W;
        int idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        float diff = input[idx] - mean;
        variance += diff * diff;
    }
    shared[threadIdx.x] = variance;
    __syncthreads();

    // Reduce variance
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    variance = shared[0] / group_size;
    __syncthreads();

    // Normalize and apply HardSwish
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c = g * channels_per_group + i / (D * H * W);
        int spatial = i % (D * H * W);
        int d = spatial / (H * W);
        int h = (spatial % (H * W)) / W;
        int w = spatial % W;
        int idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        float normalized = (input[idx] - mean) / sqrtf(variance + eps);
        float scaled = normalized * weight[c] + bias[c];
        output[idx] = hardswish(scaled);
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
    int threads = CUDA_NUM_THREADS;
    size_t shared_mem = threads * sizeof(float);
    
    group_norm_hardswish_kernel<<<blocks, threads, shared_mem>>>(
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