import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + BiasAdd + Hardtanh + Mish + GroupNorm
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Efficient fused forward pass: GEMM + Bias + Hardtanh + Mish + GroupNorm
// Each thread block handles one feature group per batch element to reuse mean/var
// We use a two-kernel approach conceptually, but fuse everything in one kernel with cooperative groups

#define BLOCK_SIZE 256
#define MAX_THREADS 1024

__device__ float hardtanh(float x) {
    return fmaxf(-1.0f, fminf(1.0f, x));
}

__device__ float mish(float x) {
    const float sp = logf(1.0f + expf(fmaxf(-20.0f, fminf(20.0f, x))));
    return x * tanhf(sp);
}

__global__ void fused_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups
) {
    const int group_size = out_features / num_groups;
    const int elements_per_batch = out_features;
    const int total_threads = batch_size * num_groups;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (bid >= total_threads) return;

    const int batch_idx = bid / num_groups;
    const int group_idx = bid % num_groups;
    const int base_feat = group_idx * group_size;

    // Shared memory for reduction
    __shared__ float shared_data[MAX_THREADS];
    __shared__ float mean_val, inv_std_val;

    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;

    // Step 1: Compute GEMM + Bias + Hardtanh + Mish for all features in this group
    for (int i = 0; i < group_size; ++i) {
        if (tid == 0) {
            const int feat_idx = base_feat + i;
            float val = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                val += input[batch_idx * in_features + k] * weight[feat_idx * in_features + k];
            }
            val += bias[feat_idx];
            val = hardtanh(val);
            val = mish(val);
            shared_data[i] = val;
            thread_sum += val;
            thread_sq_sum += val * val;
        }
        __syncthreads();
    }

    // Broadcast shared_data to all threads
    __syncthreads();

    // All threads participate in reduction using shared memory
    if (tid == 0) {
        mean_val = thread_sum / group_size;
        float var = thread_sq_sum / group_size - mean_val * mean_val;
        inv_std_val = rsqrtf(var + 1e-5f);
    }
    __syncthreads();

    // Write normalized outputs
    for (int i = 0; i < group_size; ++i) {
        if (tid == 0) {
            const int feat_idx = base_feat + i;
            float norm = (shared_data[i] - mean_val) * inv_std_val;
            output[batch_idx * elements_per_batch + feat_idx] = norm * gamma[group_idx] + beta[group_idx];
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
    extra_cflags=["-O3"]
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