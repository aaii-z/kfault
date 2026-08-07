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

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Step 1: GEMM + Subtract
    // Each thread handles one output element (batch_idx, out_idx)
    int total_elements = batch_size * out_features;
    float thread_max = -1e30f;
    float thread_sum = 0.0f;

    CUDA_1D_KERNEL_LOOP(i, total_elements) {
        int batch_idx = i / out_features;
        int out_idx = i % out_features;

        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[batch_idx * in_features + k] * weight[out_idx * in_features + k];
        }
        if (bias != nullptr) {
            sum += bias[out_idx];
        }
        sum -= subtract[out_idx];
        s_data[i] = sum;
        thread_max = fmaxf(thread_max, sum);
    }
    __syncthreads();

    // Step 2: Block-wide max reduction using shared memory
    float* shared_max = s_data;
    shared_max[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();

    // Step 3: Compute exp(x - max) and sum
    float exp_sum = 0.0f;
    CUDA_1D_KERNEL_LOOP(i, total_elements) {
        exp_sum += expf(s_data[i] - max_val);
    }

    // Block-wide sum reduction
    float* shared_sum = s_data;
    shared_sum[tid] = exp_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float total_exp_sum = shared_sum[0];
    __syncthreads();

    // Step 4: LogSumExp and GELU
    float logsumexp_val = logf(total_exp_sum) + max_val;
    float gelu_val = gelu_activation(logsumexp_val);

    // Step 5: Residual Add - Add original input back
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

    def forward(self, x):
        return fused_ops.fused_op_cuda(x, self.linear.weight, self.linear.bias, self.subtract)