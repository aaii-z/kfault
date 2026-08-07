import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + max + mean subtraction + GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__device__ float gelu_activation(float x) {
    return x * 0.5f * (1.0f + tanhf(M_SQRT2 * M_2_SQRTPI * x * (1.0f + 0.044715f * x * x)));
}

__global__ void fused_gemm_max_mean_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int max_dim
) {
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sdata[];

    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    if (bias != nullptr) {
        sum += bias[col];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reconstruct full output matrix in shared memory for this row
    __shared__ float row_buffer[8192]; // Assuming out_features <= 8192
    if (threadIdx.x < out_features) {
        row_buffer[threadIdx.x] = sdata[0];
    }
    __syncthreads();

    // Only one thread block per row computes reduction
    if (threadIdx.x == 0) {
        float max_val = -FLT_MAX;
        float mean_val = 0.0f;

        for (int i = 0; i < out_features; i++) {
            float val = row_buffer[i];
            max_val = fmaxf(max_val, val);
            mean_val += val;
        }
        mean_val /= out_features;

        // Write back GELU(output - mean) for each column
        for (int i = 0; i < out_features; i++) {
            float centered = row_buffer[i] - mean_val;
            row_buffer[i] = gelu_activation(centered);
        }
    }
    __syncthreads();

    // Each thread writes final result
    if (col < out_features) {
        output[row * out_features + col] = row_buffer[col];
    }
}
"""

fused_cpp_source = """
torch::Tensor fused_gemm_max_mean_gelu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int max_dim
);
"""

# Inline compilation
fused_extension = load_inline(
    name="fused_gemm_max_mean_gelu",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_gemm_max_mean_gelu"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + max + mean subtraction + GELU in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.max_dim = max_dim
        self.fused_op = fused_extension.fused_gemm_max_mean_gelu

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias, self.max_dim)