import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm, Min, and BiasAdd
group_norm_min_bias_add_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>

#define CUDA_NUM_THREADS 256

inline int get_num_threads(int n) {
    return (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__device__ float warp_reduce_min(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = min(val, tmp);
    }
    return val;
}

__global__ void group_norm_min_bias_add_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int num_channels,
    int num_groups,
    int channels_per_group,
    float eps) {
    
    int group_id = blockIdx.y;
    int sample_id = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float shared[];
    float* means = shared;
    float* vars = &shared[num_groups];
    
    int channel_start = group_id * channels_per_group;
    int channel_end = min(channel_start + channels_per_group, num_channels);
    
    // Compute mean and variance for the group
    float sum = 0.0f;
    float sq_sum = 0.0f;
    int count = 0;
    
    for (int c = channel_start + tid; c < channel_end; c += blockDim.x) {
        float val = input[sample_id * num_channels + c];
        sum += val;
        sq_sum += val * val;
        count++;
    }
    
    // Warp reduce sum and sq_sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sq_sum += __shfl_down_sync(0xffffffff, sq_sum, offset);
    }
    
    if (tid == 0) {
        float mean = sum / (channels_per_group);
        means[group_id] = mean;
        vars[group_id] = sq_sum / (channels_per_group) - mean * mean;
    }
    __syncthreads();
    
    // Normalize and find min
    float group_mean = means[group_id];
    float group_var = vars[group_id];
    float scale = weight[group_id];
    float shift = bias[group_id];
    
    float min_val = 1e38f;
    for (int c = channel_start + tid; c < channel_end; c += blockDim.x) {
        float val = input[sample_id * num_channels + c];
        float normalized = scale * (val - group_mean) / sqrt(group_var + eps) + shift;
        min_val = min(min_val, normalized);
    }
    
    // Warp reduce min
    min_val = warp_reduce_min(min_val);
    
    if (tid == 0) {
        output[sample_id * num_groups + group_id] = min_val;
    }
}

__global__ void bias_add_kernel(
    float* input,
    const float* bias,
    int batch_size,
    int num_channels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_channels) {
        int sample_id = idx / num_channels;
        int channel_id = idx % num_channels;
        input[idx] += bias[channel_id];
    }
}

torch::Tensor group_norm_min_bias_add_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps) {
    
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    int channels_per_group = num_channels / num_groups;
    
    auto output = torch::empty({batch_size, num_groups}, input.options());
    
    dim3 blocks(batch_size, num_groups);
    int threads = min(CUDA_NUM_THREADS, channels_per_group);
    int shared_mem_size = 2 * num_groups * sizeof(float);
    
    group_norm_min_bias_add_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        channels_per_group,
        eps);
    
    // Add the final bias
    int total_elements = batch_size * num_groups;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    bias_add_kernel<<<num_blocks, threads_per_block>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        num_groups);
    
    return output;
}
"""

group_norm_min_bias_add_cpp = """
torch::Tensor group_norm_min_bias_add_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps);
"""

# Load the custom CUDA extension
group_norm_min_bias_add = load_inline(
    name='group_norm_min_bias_add',
    cpp_sources=group_norm_min_bias_add_cpp,
    cuda_sources=group_norm_min_bias_add_source,
    functions=['group_norm_min_bias_add_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.num_groups = num_groups
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(num_groups))
        self.bias = nn.Parameter(torch.randn(bias_shape).view(-1))

    def forward(self, x):
        x = self.gemm(x)
        x = group_norm_min_bias_add.group_norm_min_bias_add_cuda(
            x, self.weight, self.bias, self.num_groups, self.eps)
        return x