import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + reduction operations with proper parallelism
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Efficient fused kernel: each block handles one sample, with multiple threads reducing
__global__ void fused_linear_sum_max_mean_logsumexp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) {

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Shared memory for reductions
    __shared__ float s_sum;
    __shared__ float s_max;
    __shared__ float s_logsumexp;
    __shared__ float s_max_local;
    __shared__ float s_acc;

    float local_sum = 0.0f;
    float local_max = -1e30f;

    // Each thread processes several output features
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        float val = bias[j];
        int offset = batch_idx * in_features;
        for (int i = 0; i < in_features; i++) {
            val += input[offset + i] * weight[j * in_features + i];
        }
        local_sum += val;
        local_max = fmaxf(local_max, val);
    }

    // Reduce sum
    __shared__ float s_sum_reduce[256];
    s_sum_reduce[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_reduce[tid] += s_sum_reduce[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        s_sum = s_sum_reduce[0];
    }
    __syncthreads();

    // Reduce max
    __shared__ float s_max_reduce[256];
    s_max_reduce[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max_reduce[tid] = fmaxf(s_max_reduce[tid], s_max_reduce[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        s_max = s_max_reduce[0];
    }
    __syncthreads();

    // Second pass: compute logsumexp = s_max + log(sum(exp(x - s_max)))
    float local_acc = 0.0f;
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        float val = bias[j];
        int offset = batch_idx * in_features;
        for (int i = 0; i < in_features; i++) {
            val += input[offset + i] * weight[j * in_features + i];
        }
        local_acc += expf(val - s_max);
    }

    __shared__ float s_acc_reduce[256];
    s_acc_reduce[tid] = local_acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_acc_reduce[tid] += s_acc_reduce[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        s_acc = s_acc_reduce[0];
    }
    __syncthreads();

    // Final logsumexp
    float logsumexp1 = s_max + logf(s_acc);
    // Second logsumexp on scalar is identity
    float result = logsumexp1;

    if (tid == 0) {
        output[batch_idx] = result;
    }
}
"""

fused_cpp_source = """
torch::Tensor fused_linear_sum_max_mean_logsumexp(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

fused_op = load_inline(
    name="fused_linear_sum_max_mean_logsumexp",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_sum_max_mean_logsumexp"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_linear_sum_max_mean_logsumexp(x, self.linear.weight, self.linear.bias)