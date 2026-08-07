import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + BiasAdd + Hardtanh + Mish + GroupNorm
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__device__ float hardtanh(float x) {
    return fmaxf(-1.0f, fminf(1.0f, x));
}

__device__ float mish(float x) {
    float exp_x = expf(fmaxf(-20.0f, fminf(20.0f, x)));
    return x * tanhf(logf(1.0f + exp_x));
}

__global__ void fused_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int group_size = out_features / num_groups;
    int elements_per_batch = out_features;

    // Shared memory for computing mean and variance per group
    extern __shared__ float shared_mem[];
    float* shared_data = shared_data = shared_mem;
    float* reduce_buf = shared_data + group_size;

    for (int g = 0; g < num_groups; g++) {
        int base_feat = g * group_size;
        float mean = 0.0f;
        float m2 = 0.0f;

        // Compute GEMM + Bias + Hardtanh + Mish for this group
        for (int i = tid; i < group_size; i += block_size) {
            int feat_idx = base_feat + i;
            float val = 0.0f;
            for (int k = 0; k < in_features; k++) {
                val += input[batch_idx * in_features + k] * weight[feat_idx * in_features + k];
            }
            val += bias[feat_idx];
            val = hardtanh(val);
            val = mish(val);
            shared_data[i] = val;
            mean += val;
            m2 += val * val;
        }

        // Reduce mean and m2 across threads
        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride && tid + stride < block_size) {
                reduce_buf[tid] = shared_data[tid] + shared_data[tid + stride];
                __syncthreads();
                shared_data[tid] = reduce_buf[tid];
            }
        }
        __syncthreads();

        float sum = (tid == 0) ? shared_data[0] : 0.0f;
        float sum_sq = (tid == 0) ? m2 : 0.0f;

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                reduce_buf[tid] = sum + sum;
                __syncthreads();
                sum = reduce_buf[tid];
            }
        }
        __syncthreads();

        if (tid == 0) {
            mean = sum / group_size;
            float var = sum_sq / group_size - mean * mean;
            float inv_std = rsqrtf(var + 1e-5f);

            for (int i = 0; i < group_size; i++) {
                int feat_idx = base_feat + i;
                float norm_val = (shared_data[i] - mean) * inv_std;
                output[batch_idx * elements_per_batch + feat_idx] = norm_val * gamma[g] + beta[g];
            }
        }
        __syncthreads();
    }
}
"""

fused_cpp_source = """
torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups
);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-use_fast_math", "-lineinfo"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + BiasAdd + Hardtanh + Mish + GroupNorm in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.num_groups = num_groups
        self.fused_forward = fused_ops.fused_forward

    def forward(self, x):
        return self.fused_forward(x, self.linear.weight, self.bias, self.groupnorm.weight, self.groupnorm.bias, self.num_groups)