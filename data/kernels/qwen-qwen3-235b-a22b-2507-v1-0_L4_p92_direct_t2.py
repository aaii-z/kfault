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

// Use tiled reduction and proper grid-stride loops
#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Hardswish activation
__device__ float hard_swish(float x) {
    float tmp = fmaxf(0.0f, fminf(3.0f, x + 3.0f));
    return x * tmp / 6.0f;
}

// Fast warp-level reduction for sum
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Fast warp-level reduction for max
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Block-level reduction for sum
__device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// Block-level reduction for max
__device__ float blockReduceMax(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// Fused GroupNorm + Tanh + HardSwish
// Each block handles one group across all spatial locations and batches
// Threads work on individual elements
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

    // Each block processes one group
    int group_id = blockIdx.x;
    if (group_id >= groups) return;

    int group_offset = group_id * elements_per_group;
    int local_idx = threadIdx.x;
    int block_size = blockDim.x;

    float sum = 0.0f, sum_sq = 0.0f;

    // Grid-stride loop over all elements in this group
    for (int i = local_idx; i < elements_per_group; i += block_size) {
        int idx = group_offset + i;
        float x = input[idx];
        sum += x;
        sum_sq += x * x;
    }

    // Compute mean and variance using Welford's online algorithm isn't efficient here, use reduction
    float mean = blockReduceSum(sum) / elements_per_group;
    float var = blockReduceSum(sum_sq) / elements_per_group - mean * mean;
    float inv_std = rsqrtf(fmaxf(var + eps, 0.0f));

    __syncthreads();

    // Now apply normalization and activations
    for (int i = local_idx; i < elements_per_group; i += block_size) {
        int idx = group_offset + i;
        float x = input[idx];
        x = (x - mean) * inv_std;

        // Get channel within group to index weight/bias
        int elem_in_group = i;
        int batch = elem_in_group / group_size;
        int residual = elem_in_group % group_size;
        int c_local = residual / (height * width);
        int c = group_id * channels_per_group + c_local;
        int offset_in_channel = residual % (height * width);
        int global_c_index = batch * num_channels * height * width + c * height * width + offset_in_channel;

        float gamma = weight[c];
        float beta = bias[c];
        x = gamma * x + beta;
        x = tanhf(x);
        x = hard_swish(x);
        output[global_c_index] = x;
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
    int idx = blockIdx.x;
    if (idx >= total_spatial) return;

    int h = (idx / width) % height;
    int w = idx % width;
    int n = idx / (height * width);

    int base_offset = n * num_channels * height * width + h * width + w;
    int stride = height * width;

    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // Use reduction to find max
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        float x = input[base_offset + c * stride];
        max_val = fmaxf(max_val, x);
    }
    max_val = blockReduceMax(max_val);

    __syncthreads();

    // Sum exp(x - max_val)
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        float x = input[base_offset + c * stride];
        sum_exp += expf(x - max_val);
    }
    sum_exp = blockReduceSum(sum_exp);

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

    // Launch one block per group
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
    dim3 grid_lse(std::min(65535, total_spatial));
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