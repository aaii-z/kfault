import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused matmul + dropout + softmax
fused_matmul_dropout_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float deviate(curandState* state) {
    return curand_uniform(state);
}

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

    // Each row has its own random state
    curandState state;
    curand_init(seed, threadIdx.x + row * blockDim.x, 0, &state);

    // Shared memory for softmax computation
    extern __shared__ float s_data[];
    float* s_input = s_data;
    float* s_max = &s_input[out_features];
    float* s_sum = &s_max[1];

    // Matrix multiplication: output[row][j] = sum_k input[row][k] * weight[j][k]
    // Compute one row of output
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[j * in_features + k];
        }
        // Add bias
        sum += bias[j];
        s_input[j] = sum;
    }
    __syncthreads();

    // Dropout: apply in-place with scaling
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        if (deviate(&state) < dropout_p) {
            s_input[j] = 0.0f;
        } else {
            s_input[j] *= scale;
        }
    }
    __syncthreads();

    // Softmax: compute max for numerical stability
    s_max[0] = s_input[0];
    for (int j = threadIdx.x + 1; j < out_features; j += blockDim.x) {
        s_max[0] = fmaxf(s_max[0], s_input[j]);
    }
    __syncthreads();

    // Reduce max across threads
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && (threadIdx.x + stride) < out_features) {
            s_max[0] = fmaxf(s_max[0], s_input[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // Broadcast max
    float max_val = s_max[0];
    __syncthreads();

    // Compute exp(x - max) and sum
    s_sum[0] = 0.0f;
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        s_input[j] = expf(s_input[j] - max_val);
        atomicAdd(&s_sum[0], s_input[j]);
    }
    __syncthreads();

    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            atomicAdd(&s_sum[0], s_input[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float sum_val = s_sum[0];
    __syncthreads();

    // Normalize: output = exp(x - max) / sum
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        output[row * out_features + j] = s_input[j] / (sum_val + 1e-6f);
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
    extra_cuda_cflags=["-O3"],
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
        self.seed = torch.randint(0, 2**63, ().item())  # Random seed for dropout
        self.scale = 1.0 / (1.0 - dropout_p)

    def forward(self, x):
        return fused_op.fused_matmul_dropout_softmax_cuda(
            x, self.weight, self.bias, self.dropout_p, self.seed, self.scale
        )