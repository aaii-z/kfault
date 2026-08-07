import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Swish Activation (x * sigmoid(x))
__global__ void swish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val / (1.0f + expf(-val));
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}

// Kernel 2: GroupNorm Statistics Calculation
__global__ void group_norm_stats_kernel(
    const float* __restrict__ input, 
    float* __restrict__ mean, 
    float* __restrict__ rstd, 
    int batch_size, int channels, int depth, int height, int width, int groups, float eps) {
    
    int n = blockIdx.y;
    int g = blockIdx.x;
    
    int channels_per_group = channels / groups;
    int spatial_size = depth * height * width;
    int count = channels_per_group * spatial_size;
    
    int c_start = g * channels_per_group;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Grid-stride loop to sum elements
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int c_offset = i / spatial_size;
        int s_offset = i % spatial_size;
        
        int c = c_start + c_offset;
        int d = s_offset / (height * width);
        int rem = s_offset % (height * width);
        int h = rem / width;
        int w = rem % width;
        
        // Input tensor is contiguous (N, C, D, H, W)
        int idx = ((n * channels + c) * depth + d) * height * width + h * width + w;
        
        float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }
    
    // Shared memory reduction
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
        float mean_val = s_sum[0] / count;
        float var_val = s_sum_sq[0] / count - mean_val * mean_val;
        mean[n * groups + g] = mean_val;
        rstd[n * groups + g] = 1.0f / sqrtf(var_val + eps);
    }
}

// Kernel 3: GroupNorm Apply + HardSwish Fusion
__global__ void group_norm_hardswish_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    const float* __restrict__ mean, 
    const float* __restrict__ rstd, 
    float* __restrict__ out,
    int batch_size, int channels, int depth, int height, int width, int groups) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * depth * height * width;
    
    if (idx < total_elements) {
        // Decompose linear index into (n, c, d, h, w)
        int w = idx % width;
        int rem = idx / width;
        int h = rem % height;
        rem /= height;
        int d = rem % depth;
        rem /= depth;
        int c = rem % channels;
        int n = rem / channels;
        
        int channels_per_group = channels / groups;
        int g = c / channels_per_group;
        
        float m = mean[n * groups + g];
        float r = rstd[n * groups + g];
        
        float val = input[idx];
        val = (val - m) * r;
        
        // Affine transform (GroupNorm)
        val = val * gamma[c] + beta[c];
        
        // HardSwish activation: x * relu6(x + 3) / 6
        float tmp = val + 3.0f;
        tmp = fmaxf(0.0f, tmp);
        tmp = fminf(6.0f, tmp);
        val = val * tmp / 6.0f;
        
        out[idx] = val;
    }
}

torch::Tensor group_norm_hardswish_cuda(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, 
    int batch_size, int channels, int depth, int height, int width, int groups, float eps) {
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto mean = torch::empty({batch_size, groups}, options);
    auto rstd = torch::empty({batch_size, groups}, options);
    auto out = torch::empty_like(input);
    
    const int block_size_stats = 256;
    dim3 grid_stats(groups, batch_size);
    
    group_norm_stats_kernel<<<grid_stats, block_size_stats>>>(
        input.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        batch_size, channels, depth, height, width, groups, eps);
    
    int total_elements = input.numel();
    const int block_size_apply = 256;
    const int num_blocks_apply = (total_elements + block_size_apply - 1) / block_size_apply;
    
    group_norm_hardswish_kernel<<<num_blocks_apply, block_size_apply>>>(
        input.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        mean.data_ptr<float>(), rstd.data_ptr<float>(), out.data_ptr<float>(),
        batch_size, channels, depth, height, width, groups);
        
    return out;
}
"""

cpp_source = """
torch::Tensor swish_cuda(torch::Tensor x);
torch::Tensor group_norm_hardswish_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int batch_size, int channels, int depth, int height, int width, int groups, float eps);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["swish_cuda", "group_norm_hardswish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.groups = groups
        self.eps = eps
        self.num_channels = out_channels
        
        # GroupNorm parameters (weight=gamma, bias=beta)
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # 1. Transposed Convolution
        x = self.conv_transpose(x)
        
        # 2. Swish Activation (Custom CUDA)
        x = custom_ops.swish_cuda(x)
        
        # 3. GroupNorm + HardSwish (Fused Custom CUDA)
        batch_size = x.size(0)
        channels = x.size(1)
        depth = x.size(2)
        height = x.size(3)
        width = x.size(4)
        
        x = custom_ops.group_norm_hardswish_cuda(
            x, self.weight, self.bias, 
            batch_size, channels, depth, height, width, 
            self.groups, self.eps
        )
        return x