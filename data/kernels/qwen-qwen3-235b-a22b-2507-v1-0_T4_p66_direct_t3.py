import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused matmul + dropout + softmax
fused_matmul_dropout_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

__global__ void fused_matmul_dropout_softmax_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float dropout_p,
    uint64_t seed,
    float scale) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Each thread block has its own random state based on row and thread index
    curandState state;
    curand_init(seed, row * blockDim.x + threadIdx.x, 0, &state);

    // Shared memory for softmax computation: [out_features] for values, 1 for max, 1 for sum
    extern __shared__ float s_data[];
    float* s_vals = s_data;
    float* s_max = &s_vals[out_features];
    float* s_sum = &s_max[1];

    // Initialize shared memory
    s_max[0] = -FLT_MAX;
    s_sum[0] = 0.0f;

    // Matrix multiplication: output[row][j] = sum_k input[row][k] * weight[j][k] + bias[j]
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[j * in_features + k];
        }
        sum += bias[j];
        s_vals[j] = sum;
    }
    __syncthreads();

    // Dropout: apply with scaling
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        float r = curand_uniform(&state);
        if (r < dropout_p) {
            s_vals[j] = 0.0f;
        } else {
            s_vals[j] *= scale;
        }
    }
    __syncthreads();

    // Softmax: find maximum value for numerical stability
    float max_val = s_vals[0];
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        max_val = fmaxf(max_val, s_vals[j]);
    }
    // Reduce within block to get max using shared memory
    if (threadIdx.x == 0) {
        s_max[0] = max_val;
    }
    __syncthreads();
    max_val = s_max[0];

    // Compute exp(x - max) and sum
    float sum_val = 0.0f;
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        float exp_val = expf(s_vals[j] - max_val);
        s_vals[j] = exp_val;
        sum_val += exp_val;
    }
    // Reduce sum in block using shared memory
    if (threadIdx.x == 0) {
        s_sum[0] = 0.0f;
    }
    __syncthreads();
    atomicAdd(s_sum, sum_val);
    __syncthreads();
    sum_val = s_sum[0];

    // Normalize and write output
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        output[row * out_features + j] = s_vals[j] / (sum_val + 1e-6f);
    }
}
"""

fused_matmul_dropout_softmax_cpp_source = """
torch::Tensor fused_matmul_dropout_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float dropout_p,
    uint64_t seed);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_matmul_dropout_softmax",
    cpp_sources=fused_matmul_dropout_softmax_cpp_source,
    cuda_sources=fused_matmul_dropout_softmax_source,
    functions=["fused_matmul_dropout_softmax_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused matmul + dropout + softmax in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.seed = torch.randint(0, 2**32, ()).item()

    def forward(self, x):
        # Ensure contiguous input
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous()

        return fused_op.fused_matmul_dropout_softmax_cuda(
            x, weight, bias, self.dropout_p, self.seed
        )