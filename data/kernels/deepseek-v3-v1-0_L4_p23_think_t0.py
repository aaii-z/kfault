import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused GroupNorm and Global Mean
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>

#define CUDA_NUM_THREADS 256
#define THREADS_PER_BLOCK 32

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (wid == 0)
        val = warpReduceSum(val);

    return val;
}

__global__ void group_norm_mean_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int num_groups,
    const float eps
) {
    const int group_size = channels / num_groups;
    const int elements_per_map = depth * height * width;
    const int elements_per_group = group_size * elements_per_map;
    const int total_elements = channels * elements_per_map;

    int batch = blockIdx.x;
    int group = blockIdx.y;
    int tid = threadIdx.x;

    const float* group_input = input + batch * total_elements + group * elements_per_group;
    extern __shared__ float shared[];

    float* group_vals = shared;
    float* mean_vals = shared + elements_per_group;
    float* var_vals = mean_vals + 1;

    if (tid == 0) {
        mean_vals[0] = 0.0f;
        var_vals[0] = 0.0f;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        float val = group_input[i];
        sum += val;
        group_vals[i] = val;
    }
    sum = blockReduceSum(sum);
    if (tid == 0) {
        mean_vals[0] = sum / elements_per_group;
    }
    __syncthreads();

    float mean = mean_vals[0];
    float sum_sq = 0.0f;
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        float diff = group_vals[i] - mean;
        sum_sq += diff * diff;
    }
    sum_sq = blockReduceSum(sum_sq);
    if (tid == 0) {
        var_vals[0] = sum_sq / elements_per_group + eps;
    }
    __syncthreads();

    float inv_std = rsqrtf(var_vals[0]);
    int channel_start = group * group_size;

    float group_sum = 0.0f;
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        int channel_in_group = i / elements_per_map;
        int channel = channel_start + channel_in_group;
        int spatial_idx = i % elements_per_map;
        
        float val = group_vals[i];
        float normalized = (val - mean) * inv_std;
        float scaled = normalized * weight[channel] + bias[channel];
        
        group_sum += scaled;
    }
    group_sum = blockReduceSum(group_sum);

    if (tid == 0) {
        atomicAdd(output + batch, group_sum / total_elements);
    }
}

torch::Tensor fused_group_norm_mean_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps = 1e-5
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto output = torch::zeros({batch_size}, input.options());
    
    dim3 grid(batch_size, num_groups);
    int threads = opt_n_threads(channels / num_groups * depth * height * width);
    size_t shared_memory = (channels / num_groups * depth * height * width + 2) * sizeof(float);
    
    group_norm_mean_kernel<<<grid, threads, shared_memory>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        num_groups,
        eps
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor fused_group_norm_mean_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps);"

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_group_norm_mean_cuda'],
    verbose=True,
    extra_cflags=['-O2'],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.num_groups = num_groups
        self.group_norm_weight = nn.Parameter(torch.ones(out_channels))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops.fused_group_norm_mean_cuda(
            x, 
            self.group_norm_weight, 
            self.group_norm_bias, 
            self.num_groups
        )
        return x