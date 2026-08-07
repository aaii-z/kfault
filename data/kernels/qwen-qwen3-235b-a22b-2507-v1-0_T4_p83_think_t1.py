import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_gn_min_clamp_dropout_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void compute_group_stats_kernel(
    const float* input, 
    float* means, 
    float* vars,
    int batch_size,
    int channels_per_group,
    int depth,
    int height,
    int width,
    int groups,
    int total_elements_per_group) {
    
    const int group_id = blockIdx.x;
    const int num_threads = blockDim.x;
    const int thread_id = threadIdx.x;
    
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    const int elements_per_thread = (total_elements_per_group + num_threads - 1) / num_threads;
    const int start_idx = thread_id * elements_per_thread;
    const int end_idx = min((thread_id + 1) * elements_per_thread, total_elements_per_group);
    
    const int group_offset = group_id * total_elements_per_group;
    
    for (int i = start_idx; i < end_idx; ++i) {
        float val = input[group_offset + i];
        sum += val;
        sq_sum += val * val;
    }
    
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_sq_sum[BLOCK_SIZE];
    
    shared_sum[thread_id] = sum;
    shared_sq_sum[thread_id] = sq_sum;
    __syncthreads();
    
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_sum[thread_id] += shared_sum[thread_id + stride];
            shared_sq_sum[thread_id] += shared_sq_sum[thread_id + stride];
        }
        __syncthreads();
    }
    
    if (thread_id == 0) {
        const float mean = shared_sum[0] / total_elements_per_group;
        const float var = fmaxf(shared_sq_sum[0] / total_elements_per_group - mean * mean, 0.0f);
        means[group_id] = mean;
        vars[group_id] = var;
    }
}

__global__ void fused_gn_min_clamp_dropout_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const float* means,
    const float* vars,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    int groups,
    int channels_per_group,
    float min_val,
    float max_val,
    float dropout_p,
    bool training,
    uint64_t seed,
    float eps) {
    
    const int total_elements = batch_size * channels * depth * height * width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    const int w = idx % width;
    const int h = (idx / width) % height;
    const int d = (idx / (width * height)) % depth;
    const int c = (idx / (width * height * depth)) % channels;
    const int b = idx / (width * height * depth * channels);
    
    const int group_id = c / channels_per_group;
    const int channel_in_group = c % channels_per_group;
    
    const float mean = means[group_id];
    const float inv_var = 1.0f / sqrtf(vars[group_id] + eps);
    
    const float x = input[idx];
    const float normalized = (x - mean) * inv_var * weight[c] + bias[c];
    
    float x_after_min = (normalized < min_val) ? normalized : min_val;
    float x_after_clamp = (x_after_min < min_val) ? min_val : (x_after_min > max_val) ? max_val : x_after_min;
    
    if (training) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float r = curand_uniform(&state);
        if (r < dropout_p) {
            x_after_clamp = 0.0f;
        } else {
            x_after_clamp = x_after_clamp / (1.0f - dropout_p);
        }
    }
    
    output[idx] = x_after_clamp;
}

torch::Tensor fused_gn_min_clamp_dropout_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float min_val,
    float max_val,
    float dropout_p,
    bool training,
    float eps,
    uint64_t seed) {
    
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto depth = input.size(2);
    const auto height = input.size(3);
    const auto width = input.size(4);
    const int channels_per_group = channels / groups;
    const int total_elements = input.numel();
    const int total_elements_per_group = batch_size * channels_per_group * depth * height * width;
    
    auto means = torch::empty(groups, input.options());
    auto vars = torch::empty(groups, input.options());
    
    const int num_stats_blocks = groups;
    compute_group_stats_kernel<<<num_stats_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        means.data_ptr<float>(),
        vars.data_ptr<float>(),
        batch_size,
        channels_per_group,
        depth,
        height,
        width,
        groups,
        total_elements_per_group
    );
    
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_gn_min_clamp_dropout_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        means.data_ptr<float>(),
        vars.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        groups,
        channels_per_group,
        min_val,
        max_val,
        dropout_p,
        training,
        seed,
        eps
    );
    
    return output;
}
"""

fused_gn_min_clamp_dropout_cpp = """
torch::Tensor fused_gn_min_clamp_dropout_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float min_val,
    float max_val,
    float dropout_p,
    bool training,
    float eps,
    uint64_t seed);
"""

fused_gn_min_clamp_dropout = load_inline(
    name="fused_gn_min_clamp_dropout",
    cpp_sources=fused_gn_min_clamp_dropout_cpp,
    cuda_sources=fused_gn_min_clamp_dropout_source,
    functions=["fused_gn_min_clamp_dropout_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-maxrregcount=255"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels, affine=True)
        self.dropout_p = dropout_p
        self.min_value = min_value
        self.max_value = max_value
        self.groups = groups
        self.eps = self.norm.eps
        self.fused_op = fused_gn_min_clamp_dropout

    def forward(self, x):
        x = self.conv(x)
        if self.training:
            seed = torch.cuda.initial_seed() if self.training else 0
            x = self.fused_op.fused_gn_min_clamp_dropout_cuda(
                x, 
                self.norm.weight, 
                self.norm.bias, 
                self.groups, 
                self.min_value, 
                self.max_value, 
                self.dropout_p, 
                True, 
                self.eps,
                seed
            )
        else:
            x = self.norm(x)
            x = torch.min(x, torch.tensor(self.min_value, device=x.device))
            x = torch.clamp(x, min=self.min_value, max=self.max_value)
            x = nn.functional.dropout(x, p=self.dropout_p, training=self.training)
        return x