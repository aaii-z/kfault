import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused GroupNorm + Tanh + HardSwish with proper parallelization
fused_gn_act_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>

#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float hard_swish(float x) {
    float tmp = fmaxf(0.0f, fminf(3.0f, x + 3.0f));
    return x * tmp / 6.0f;
}

// Fast warp-level mean and variance reduction
__device__ void warp_reduce(volatile float* data, int tid) {
    if (tid < 32) {
        float val = data[tid];
        for (int offset = 1; offset < 32; offset *= 2)
            val = val + __shfl_down_sync(0xffffffff, val, offset);
        data[0] = val;
    }
}

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

    const int channels_per_group = num_channels / groups;
    const int group_size = channels_per_group * height * width;
    const int total_elements = batch_size * num_channels * height * width;
    const int spatial_size = height * width;

    // Each block handles one group instance (per batch and spatial location)
    int batch_id = blockIdx.y;
    int hw_id = blockIdx.z;
    int group_id = blockIdx.x;

    if (batch_id >= batch_size || hw_id >= spatial_size || group_id >= groups) return;

    const int group_start = group_id * channels_per_group;
    const int h = hw_id / width;
    const int w = hw_id % width;

    const float* input_base = input + batch_id * num_channels * spatial_size + group_start * spatial_size + h * width + w;
    float* output_base = output + batch_id * num_channels * spatial_size + group_start * spatial_size + h * width + w;

    // Shared memory for mean and variance reduction
    __shared__ float shared_data[256]; // assume max 256 threads per block
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;

    int tid = threadIdx.x;
    int c = tid;

    // Load data and compute local sum and sum of squares
    if (c < channels_per_group) {
        float val = input_base[c * spatial_size];
        thread_sum = val;
        thread_sq_sum = val * val;
    }

    // Reduce within block using shared memory
    int block_size = blockDim.x;
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        shared_data[tid] = thread_sum;
        __syncthreads();
        if (tid < stride) {
            thread_sum += shared_data[tid + stride];
            shared_data[tid] = thread_sq_sum;
            __syncthreads();
            thread_sq_sum += shared_data[tid + stride];
        }
    }

    // Final reduction at thread 0
    float mean, var;
    if (tid == 0) {
        mean = thread_sum / channels_per_group;
        var = thread_sq_sum / channels_per_group - mean * mean;
    } else {
        mean = 0.0f;
        var = 0.0f;
    }

    // Broadcast mean and variance
    mean = __shfl_sync(0xffffffff, mean, 0);
    var = __shfl_sync(0xffffffff, var, 0);
    float inv_std = rsqrtf(var + eps);

    // Apply GroupNorm, Tanh, HardSwish
    if (c < channels_per_group) {
        float x = input_base[c * spatial_size];
        x = (x - mean) * inv_std;
        x = weight[group_start + c] * x + bias[group_start + c];
        x = tanhf(x);
        x = hard_swish(x);
        output_base[c * spatial_size] = x;
    }
}

// Optimized LogSumExp kernel with block per spatial location
__global__ void logsumexp_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_channels,
    int height,
    int width) {

    int spatial_id = blockIdx.x;
    int batch_id = blockIdx.y;

    if (spatial_id >= height * width || batch_id >= batch_size) return;

    const int offset_base = batch_id * num_channels * height * width + spatial_id;
    const int stride = height * width;

    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // Find max in parallel using warp shuffle
    float local_max = -INFINITY;
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        local_max = fmaxf(local_max, input[offset_base + c * stride]);
    }

    // Warp-level reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax((int*)&reinterpret_cast<volatile int&>(max_val), __float_as_int(local_max));
    }
    __syncthreads();

    // Broadcast max (simplified: recompute max across threads)
    max_val = local_max;
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    // Sum exp(x_i - max)
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        local_sum += expf(input[offset_base + c * stride] - max_val);
    }

    // Reduce sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&sum_exp, local_sum);
    }
    __syncthreads();

    // Only thread 0 writes result
    if (threadIdx.x == 0) {
        output[batch_id * height * width + spatial_id] = max_val + logf(sum_exp);
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
    int spatial_size = height * width;
    int channels_per_group = num_channels / groups;

    auto output = torch::empty_like(input);

    dim3 block(channels_per_group);
    dim3 grid(groups, batch_size, spatial_size);

    fused_gn_tanh_hardswish_kernel<<<grid, block, 0, stream>>>(
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
    int spatial_size = height * width;

    auto output = torch::empty({batch_size, 1, height, width}, input.options());

    dim3 block(std::min(256, num_channels));
    dim3 grid(spatial_size, batch_size);

    logsumexp_kernel<<<grid, block, 0, stream>>>(
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
    cuda_sources=fused_gn_act_source,
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