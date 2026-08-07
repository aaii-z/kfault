import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Gemm + LogSumExp + 2x LeakyReLU + 2x GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

__device__ inline float leaky_relu(float x, float negative_slope) {
    return fmaxf(x, negative_slope * x);
}

__global__ void fused_gemm_lse_lrelu_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features)
{
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int feature_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for input row
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;

    // Load input row into shared memory
    for (int k = tid; k < in_features; k += blockDim.x) {
        s_input[k] = input[row * in_features + k];
    }
    __syncthreads();

    float element = 0.0f;
    if (feature_id < out_features) {
        // Compute GEMM: output[row][feature_id] = dot(input[row], weight[feature_id]) + bias[feature_id]
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += s_input[k] * weight[feature_id * in_features + k];
        }
        element = sum + bias[feature_id];

        // Write the full output of Gemm to shared memory (all features)
        s_input[tid] = element;
    }
    __syncthreads();

    // Now perform LogSumExp reduction across out_features for this row
    // Use block-wide reduction to compute max and sum(exp(x - max))
    float max_val = -INFINITY;
    if (tid < out_features) {
        max_val = (tid < out_features && feature_id < out_features) ? s_input[tid] : -INFINITY;
    }
    __syncthreads();

    // Reduce to find max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < out_features) {
            max_val = fmaxf(max_val, s_input[tid + stride]);
        }
        __syncthreads();
    }

    // Broadcast max_val to all threads
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = max_val;
    }
    __syncthreads();
    max_val = shared_max;

    // Compute sum of exps
    float sum_exp = 0.0f;
    if (feature_id < out_features) {
        sum_exp = expf(s_input[tid] - max_val);
    }

    // Reduce sum_exp
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, stride);
    }

    // First thread computes logsumexp
    float lse = 0.0f;
    if (threadIdx.x == 0) {
        lse = max_val + logf(sum_exp);
    }
    __syncthreads();

    // Broadcast lse to all threads
    __shared__ float logsumexp_val;
    if (threadIdx.x == 0) {
        logsumexp_val = lse;
    }
    __syncthreads();
    lse = logsumexp_val;

    // Apply two LeakyReLU and two GELU activations
    float activated = lse;
    activated = leaky_relu(activated, 0.01f);
    activated = leaky_relu(activated, 0.01f);
    activated = gelu(activated);
    activated = gelu(activated);

    // Only thread 0 writes the final output for this row
    if (threadIdx.x == 0) {
        output[row] = activated;
    }
}
"""

fused_cpp_source = """
torch::Tensor fused_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Gemm + LogSumExp + LeakyReLU + GELU operations in a single CUDA kernel.
    Output shape is (batch_size, 1) to match the original model.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_kernel = fused_op.fused_forward

    def forward(self, x):
        # Launch kernel with grid and block dimensions
        batch_size = x.size(0)
        out_features = self.linear.out_features
        block_size = min(1024, out_features)
        grid = (min(32, (out_features + block_size - 1) // block_size), batch_size)
        shared_mem_size = block_size * sizeof(float)  # for input + temp storage
        output = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)
        self.fused_kernel.fused_forward(x, self.linear.weight, self.linear.bias, output, grid=grid, block=block_size, shared=shared_mem_size)
        return output