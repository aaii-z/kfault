import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Gemm + LogSumExp + 2x LeakyReLU + 2x GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(M_SQRT2 * M_2_SQRTPI * x * (1.0f + 0.044715f * x * x)));
}

__device__ inline float leaky_relu(float x, float negative_slope) {
    return fmaxf(x, x * negative_slope);
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
    int bid = blockIdx.x;

    extern __shared__ float shared_data[];
    float* shared_mem = shared_data;
    float* shared_sum = &shared_data[256];

    float sum = 0.0f;

    // Matrix multiplication: C[row] = A[row,:] * B
    for (int k = tid; k < in_features; k += 256) {
        sum += input[row * in_features + k] * weight[bid * in_features + k];
    }

    shared_mem[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    float result = shared_mem[0];
    if (tid == 0) {
        result += bias[bid];
        shared_sum[bid] = result;
    }
    __syncthreads();

    // LogSumExp: compute max and sum of exps
    if (bid == 0 && tid == 0) {
        float max_val = shared_sum[0];
        for (int i = 1; i < out_features; i++) {
            max_val = fmaxf(max_val, shared_sum[i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < out_features; i++) {
            sum_exp += expf(shared_sum[i] - max_val);
        }

        result = max_val + logf(sum_exp);
        shared_sum[0] = result;
    }
    __syncthreads();

    // Apply two LeakyReLU and two GELU activations
    if (bid == 0) {
        float val = shared_sum[0];
        val = leaky_relu(val, 0.01f);
        val = leaky_relu(val, 0.01f);
        val = gelu(val);
        val = gelu(val);

        if (tid == 0) {
            output[row * 1] = val;  // Output is (batch_size, 1)
        }
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
        output = self.fused_kernel(x, self.linear.weight, self.linear.bias)
        return output.unsqueeze(-1)  # Reshape to (batch_size, 1) to match expected output