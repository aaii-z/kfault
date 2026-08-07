import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_NUM_THREADS 256

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val, T* shared, int warps_per_block) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    if (threadIdx.x < warps_per_block) {
        val = shared[threadIdx.x];
    } else {
        val = 0;
    }

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
    extern __shared__ float shared[];
    int warps_per_block = (blockDim.x + 31) / 32;
    float* group_means = shared;
    float* group_vars = shared + num_groups;
    float* reduction_buffer = shared + 2 * num_groups;

    int spatial_size = depth * height * width;
    int group_size = channels / num_groups;
    int elements_per_group = group_size * spatial_size;
    int total_elements = channels * spatial_size;

    int batch = blockIdx.x;
    int tid = threadIdx.x;

    const float* batch_input = input + batch * total_elements;

    if (tid < num_groups * 2) {
        shared[tid] = 0.0f;
    }
    __syncthreads();

    for (int group = 0; group < num_groups; group++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        const float* group_input = batch_input + group * elements_per_group;
        
        for (int i = tid; i < elements_per_group; i += blockDim.x) {
            float val = group_input[i];
            sum += val;
            sum_sq += val * val;
        }

        sum = blockReduceSum(sum, reduction_buffer, warps_per_block);
        sum_sq = blockReduceSum(sum_sq, reduction_buffer, warps_per_block);

        if (tid == 0) {
            float mean = sum / elements_per_group;
            float variance = (sum_sq / elements_per_group) - (mean * mean);
            group_means[group] = mean;
            group_vars[group] = variance + eps;
        }
        __syncthreads();
    }

    float global_sum = 0.0f;
    for (int i = tid; i < total_elements; i += blockDim.x) {
        int channel = i / spatial_size;
        int group = channel / group_size;
        int channel_in_group = channel % group_size;
        
        float val = batch_input[i];
        float mean = group_means[group];
        float inv_std = rsqrtf(group_vars[group]);
        float normalized = (val - mean) * inv_std;
        float scaled = normalized * weight[channel] + bias[channel];
        global_sum += scaled;
    }

    global_sum = blockReduceSum(global_sum, reduction_buffer, warps_per_block);

    if (tid == 0) {
        output[batch] = global_sum / total_elements;
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
    
    int threads = min(channels * depth * height * width, 1024);
    int warps_per_block = (threads + 31) / 32;
    size_t shared_mem_size = (2 * num_groups + warps_per_block) * sizeof(float);
    
    group_norm_mean_kernel<<<batch_size, threads, shared_mem_size>>>(
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