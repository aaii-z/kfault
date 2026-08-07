import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized CUDA kernels with proper parallelism and memory coalescing
fused_gn_act_logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <algorithm>

// Hardswish activation
__device__ float hard_swish(float x) {
    float tmp = fmaxf(0.0f, fminf(3.0f, x + 3.0f));
    return x * tmp / 6.0f;
}

// Warp-level sum reduction
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Warp-level max reduction
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Fused GroupNorm + Tanh + HardSwish kernel
// Each block handles one group, threads process elements in that group
__global__ void fused_gn_tanh_hardswish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int num_channels,
    int height,
    int width,
    int groups,
    float eps) {

    int channels_per_group = num_channels / groups;
    int group_size = channels_per_group * height * width;
    int total_elements = batch_size * num_channels * height * width;
    int elements_per_group = batch_size * group_size;

    int group_id = blockIdx.x;
    if (group_id >= groups) return;

    int group_offset = group_id * elements_per_group;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float sum = 0.0f, sum_sq = 0.0f;

    // Compute sum and sum of squares for this group
    for (int i = tid; i < elements_per_group; i += block_size) {
        int idx = group_offset + i;
        float x = input[idx];
        sum += x;
        sum_sq += x * x;
    }

    // Block-wide reduction for sum and sum_sq
    float global_sum = 0.0f, global_sum_sq = 0.0f;
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_sum[0] / elements_per_group;
        float var = s_sum_sq[0] / elements_per_group - mean * mean;
        float inv_std = rsqrtf(fmaxf(var + eps, 1e-12f));

        // Store mean and inv_std in shared memory for activation phase
        s_sum[0] = mean;
        s_sum_sq[0] = inv_std;
    }
    __syncthreads();

    float mean = s_sum[0];
    float inv_std = s_sum_sq[0];

    // Apply GroupNorm, Tanh, HardSwish
    for (int i = tid; i < elements_per_group; i += block_size) {
        int idx = group_offset + i;
        float x = input[idx];
        x = (x - mean) * inv_std;

        // Map linear index to channel and spatial location
        int elem_in_group = i;
        int c_local = (elem_in_group % group_size) / (height * width);
        int c = group_id * channels_per_group + c_local;

        float gamma = weight[c];
        float beta = bias[c];
        x = gamma * x + beta;
        x = tanhf(x);
        x = hard_swish(x);

        output[idx] = x;
    }
}

// Fused LogSumExp kernel: each block handles one (n, h, w)
__global__ void logsumexp_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_channels,
    int height,
    int width) {

    int total_spatial = batch_size * height * width;
    int spatial_id = blockIdx.x;
    if (spatial_id >= total_spatial) return;

    int w = spatial_id % width;
    int h = (spatial_id / width) % height;
    int n = spatial_id / (height * width);

    int base_offset = n * num_channels * height * width + h * width + w;
    int stride = height * width;

    float max_val = -INFINITY;

    // Find max over channel dimension
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        float x = input[base_offset + c * stride];
        max_val = fmaxf(max_val, x);
    }
    max_val = warpReduceMax(max_val);
    if (threadIdx.x % 32 == 0) {
        ((float*)shfl_sync(0xffffffff, __float_as_int(max_val), 0))[blockIdx.x % 32] = max_val;
    }
    __syncthreads();

    // Use shared memory to collect warp maxes
    __shared__ float shared_max[32];
    if (threadIdx.x < 32) {
        shared_max[threadIdx.x] = -INFINITY;
    }
    __syncthreads();

    if (threadIdx.x % 32 == 0) {
        atomicMax((int*)&shared_max[threadIdx.x / 32], __float_as_int(max_val));
    }
    __syncthreads();

    max_val = shared_max[0];
    for (int i = 1; i < 32 && i < (blockDim.x + 31) / 32; i++) {
        max_val = fmaxf(max_val, shared_max[i]);
    }

    __syncthreads();

    float sum_exp = 0.0f;
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        float x = input[base_offset + c * stride];
        sum_exp += expf(x - max_val);
    }
    sum_exp = warpReduceSum(sum_exp);
    __syncthreads();

    // Use shared memory for sum reduction
    __shared__ float shared_sum[32];
    if (threadIdx.x < 32) {
        shared_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&shared_sum[threadIdx.x / 32], sum_exp);
    }
    __syncthreads();

    sum_exp = shared_sum[0];
    for (int i = 1; i < 32 && i < (blockDim.x + 31) / 32; i++) {
        sum_exp += shared_sum[i];
    }

    if (threadIdx.x == 0) {
        output[n * height * width + h * width + w] = max_val + logf(sum_exp);
    }
}

torch::Tensor fused_gn_tanh_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float eps) {
    
    auto stream = at::cuda::getCurrentCUDAStream();

    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    dim3 grid_gn(groups);
    dim3 block_gn(threads_per_block);

    fused_gn_tanh_hardswish_kernel<<<grid_gn, block_gn, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, num_channels, height, width, groups, eps
    );

    return output;
}

torch::Tensor logsumexp_cuda(torch::Tensor input) {
    auto stream = at::cuda::getCurrentCUDAStream();

    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty({batch_size, 1, height, width}, input.options());

    const int threads_per_block = 256;
    int total_spatial = batch_size * height * width;
    int num_blocks = std::min(65535, total_spatial);
    dim3 grid_lse(num_blocks);
    dim3 block_lse(threads_per_block);

    logsumexp_kernel<<<grid_lse, block_lse, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, num_channels, height, width
    );

    return output;
}
"""

fused_gn_act_cpp = """
torch::Tensor fused_gn_tanh_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int groups,
    float eps);

torch::Tensor logsumexp_cuda(torch::Tensor input);
"""

# Compile the inline CUDA kernels
fused_ops = load_inline(
    name="fused_gn_act",
    cpp_sources=fused_gn_act_cpp,
    cuda_sources=fused_gn_act_logsumexp_source,
    functions=["fused_gn_tanh_hardswish_cuda", "logsumexp_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernels for GroupNorm + Tanh + HardSwish and LogSumExp.
    Convolution remains in PyTorch for correctness and compatibility.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.fused_gn_act = fused_ops.fused_gn_tanh_hardswish_cuda
        self.fused_logsumexp = fused_ops.logsumexp_cuda

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Fused GroupNorm + Tanh + HardSwish
        x_act = self.fused_gn_act(
            x_conv, self.group_norm.weight, self.group_norm.bias,
            self.groups, self.eps
        )
        # Residual Addition
        x_res = x_conv + x_act
        # LogSumExp
        x_logsumexp = self.fused_logsumexp(x_res)
        return x_logsumexp