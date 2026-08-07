import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Gemm + LogSumExp + 2x LeakyReLU + 2x GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Approximate GELU function
__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

// Leaky ReLU
__device__ inline float leaky_relu(float x, float negative_slope) {
    return x >= 0.0f ? x : x * negative_slope;
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
    int row = blockIdx.x;  // one block per batch element
    int tid = threadIdx.x;
    int feature_id = blockIdx.y;  // feature index for matmul

    extern __shared__ float shared_mem[];
    float* s_data = shared_mem;
    float* s_sum = &shared_mem[blockDim.x];

    // Step 1: Compute linear transformation: y[feature_id] = W[feature_id] @ x + b[feature_id]
    float sum = 0.0f;
    for (int k = tid; k < in_features; k += blockDim.x) {
        sum += input[row * in_features + k] * weight[feature_id * in_features + k];
    }
    s_data[tid] = sum;
    __syncthreads();

    // Reduce within block to get full dot product
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 stores the result
    float result = 0.0f;
    if (tid == 0) {
        result = s_data[0] + bias[feature_id];
        s_sum[feature_id] = result;
    }
    __syncthreads();

    // Only one thread block (blockIdx.y == 0) performs the logsumexp and activations
    if (feature_id == 0 && tid == 0) {
        // Find max value for numerical stability
        float max_val = s_sum[0];
        for (int i = 1; i < out_features; i++) {
            float val = s_sum[i];
            max_val = fmaxf(max_val, val);
        }

        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int i = 0; i < out_features; i++) {
            sum_exp += expf(s_sum[i] - max_val);
        }

        // LogSumExp: max_val + log(sum_exp)
        float lse = max_val + logf(sum_exp);

        // Apply two LeakyReLU and two GELU activations
        float act = leaky_relu(lse, 0.01f);
        act = leaky_relu(act, 0.01f);
        act = gelu(act);
        act = gelu(act);

        output[row] = act;
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
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Gemm + LogSumExp + LeakyReLU + GELU operations in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_kernel = fused_op.fused_forward

    def forward(self, x):
        return self.fused_kernel(x, self.linear.weight, self.linear.bias)