import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused GroupNorm + Tanh + HardSwish with proper parallelization and reduced divergence
fused_gn_act_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float hard_swish(float x) {
    float tmp = fmaxf(0.0f, fminf(3.0f, x + 3.0f));
    return x * tmp / 6.0f;
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
    const int spatial_size = height * width;
    const int total_elements = batch_size * num_channels * spatial_size;
    const int group_size = channels_per_group * spatial_size;

    // Each thread processes one element
    CUDA_1D_KERNEL_LOOP(idx, total_elements) {
        int spatial_idx = idx % spatial_size;
        int c = (idx / spatial_size) % num_channels;
        int batch = idx / (num_channels * spatial_size);

        int group_id = c / channels_per_group;
        int c_local = c % channels_per_group;

        int group_offset = group_id * group_size;
        int channel_offset = c_local * spatial_size;

        // Compute mean and variance for the group
        float mean = 0.0f, var = 0.0f;
        const float* group_base = input + batch * num_channels * spatial_size + group_offset + spatial_idx;

        // Use cooperative grid-wide reduction for mean
        for (int cc = 0; cc < channels_per_group; ++cc) {
            float val = group_base[cc * spatial_size];
            mean += val;
        }
        mean /= channels_per_group;

        for (int cc = 0; cc < channels_per_group; ++cc) {
            float val = group_base[cc * spatial_size];
            float diff = val - mean;
            var += diff * diff;
        }
        var /= channels_per_group;
        float inv_std = rsqrtf(var + eps);

        // Apply GroupNorm, Tanh, HardSwish
        float x = input[idx];
        x = (x - mean) * inv_std;
        x = weight[c] * x + bias[c];
        x = tanhf(x);
        x = hard_swish(x);
        output[idx] = x;
    }
}

// Optimized LogSumExp kernel using block-wide reduction with warp shuffle
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

    // Find maximum value using warp shuffle
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        float val = input[offset_base + c * stride];
        max_val = fmaxf(max_val, val);
    }

    // Warp-level reduction for max
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    // Broadcast max from thread 0 of warp 0
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // Sum exp(x_i - max)
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < num_channels; c += blockDim.x) {
        local_sum += expf(input[offset_base + c * stride] - max_val);
    }

    // Reduce sum within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Only one thread per warp accumulates, then one per block
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&sum_exp, local_sum);
    }
    __syncthreads();

    // Final write by thread 0
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

    auto output = torch::empty_like(input);

    const int total_elements = batch_size * num_channels * height * width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    dim3 block(block_size);
    dim3 grid(grid_size);

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