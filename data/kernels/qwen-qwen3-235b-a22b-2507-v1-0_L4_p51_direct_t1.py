import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and Residual Add
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_gemm_subtract_globalavgpool_logsumexp_gelu_residual_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* subtract,
    float* output,
    int batch_size,
    int in_features,
    int out_features) {

    extern __shared__ float s_data[];

    float thread_sum = 0.0f;
    float thread_max = -1e30f;

    int total_gemms = batch_size * out_features;
    CUDA_1D_KERNEL_LOOP(i, total_gemms) {
        int batch_idx = i / out_features;
        int out_idx = i % out_features;

        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[batch_idx * in_features + k] * weight[out_idx * in_features + k];
        }
        if (bias) {
            sum += bias[out_idx];
        }
        sum -= subtract[out_idx];
        s_data[i] = sum;
        thread_max = fmaxf(thread_max, sum);
    }
    __syncthreads();

    // Use shared memory for max reduction
    float max_val = thread_max;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, stride));
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    __syncthreads();

    // Compute exp(x - max) and sum
    float exp_sum = 0.0f;
    CUDA_1D_KERNEL_LOOP(i, total_gemms) {
        exp_sum += expf(s_data[i] - max_val);
    }

    // Reduce exp_sum across threads
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        exp_sum += __shfl_down_sync(0xffffffff, exp_sum, stride);
    }
    exp_sum = __shfl_sync(0xffffffff, exp_sum, 0);
    __syncthreads();

    float logsumexp_val = logf(exp_sum) + max_val;
    float gelu_val = gelu_activation(logsumexp_val);

    // Final residual add: (B, 1) broadcast added to (B, in_features)
    CUDA_1D_KERNEL_LOOP(i, batch_size * in_features) {
        output[i] = gelu_val + input[i];
    }
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.fused_op = fused_ops.fused_op_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias, self.subtract)