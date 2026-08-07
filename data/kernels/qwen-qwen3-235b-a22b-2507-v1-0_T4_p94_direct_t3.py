import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + BiasAdd + Hardtanh + Mish + GroupNorm
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

__device__ float hardtanh(float x) {
    return fmaxf(-1.0f, fminf(1.0f, x));
}

__device__ float mish(float x) {
    const float threshold = 20.0f;
    float exp_x = expf(fmaxf(-threshold, fminf(threshold, x)));
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
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int group_id = blockIdx.x;
    int block_size = blockDim.x;

    int group_size = out_features / num_groups;
    int base_feat = group_id * group_size;

    // Shared memory for input and partial sums
    extern __shared__ float shared_mem[];
    float* input_shared = shared_mem;
    float* output_shared = &shared_mem[in_features];
    float* reduce_buf = &shared_mem[in_features + group_size];

    // Load input for this batch
    if (threadIdx.y == 0) {
        for (int i = tid; i < in_features; i += block_size) {
            input_shared[i] = input[batch_idx * in_features + i];
        }
    }
    __syncthreads();

    // Compute GEMM + Bias + Hardtanh + Mish for each feature in the group
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = tid; i < group_size; i += block_size) {
        int feat_idx = base_feat + i;
        float val = 0.0f;
        for (int k = 0; k < in_features; k++) {
            val += input_shared[k] * weight[feat_idx * in_features + k];
        }
        val += bias[feat_idx];
        val = hardtanh(val);
        val = mish(val);
        output_shared[i] = val;
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Block-wide reduction for mean and variance
    float* cache = &shared_mem[in_features + group_size + block_size];
    int cache_idx = threadIdx.y * blockDim.x + threadIdx.x;

    cache[cache_idx] = local_sum;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[cache_idx] += cache[cache_idx + s];
        }
        __syncthreads();
    }
    float mean = (tid == 0) ? cache[threadIdx.y * blockDim.x] / group_size : 0.0f;

    cache[cache_idx] = local_sum_sq;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[cache_idx] += cache[cache_idx + s];
        }
        __syncthreads();
    }
    float mean_sq = (tid == 0) ? cache[threadIdx.y * blockDim.x] / group_size : 0.0f;

    float var = (tid == 0) ? mean_sq - mean * mean : 0.0f;
    float inv_std = (tid == 0) ? rsqrtf(var + 1e-5f) : 0.0f;
    __syncthreads();

    // Normalize and apply affine transform
    if (tid == 0) {
        float g = gamma[group_id];
        float b = beta[group_id];
        for (int i = 0; i < group_size; i++) {
            int feat_idx = base_feat + i;
            float norm_val = (output_shared[i] - mean) * inv_std;
            output[batch_idx * out_features + feat_idx] = norm_val * g + b;
        }
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

def fused_forward(input, weight, bias, gamma, beta, num_groups):
    batch_size = input.size(0)
    out_features = weight.size(0)
    in_features = weight.size(1)

    output = torch.empty(batch_size, out_features, device=input.device, dtype=input.dtype)

    # Use 2D grid: (num_groups, batch_size)
    grid = (num_groups, batch_size)
    block_size = 256
    block = (block_size, 1, 1)

    # Shared memory: input (in_features) + output_shared (group_size) + reduce_buf (2 * block_size)
    group_size = out_features // num_groups
    smem_size = (in_features + group_size + 3 * block_size) * sizeof(float)

    fused_forward_kernel<<<grid, block, smem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups
    );

    return output

# Register the function
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=[fused_cpp_source, "torch::Tensor fused_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor gamma, torch::Tensor beta, int num_groups) { return fused_forward(input, weight, bias, gamma, beta, num_groups); }"],
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