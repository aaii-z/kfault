import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + max reduction along dim=-1 (dim 1) + mean subtraction + GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__device__ float gelu_activation(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float tanh_arg = sqrt_2_over_pi * x * (1.0f + coeff * x * x);
    return x * 0.5f * (1.0f + tanhf(tanh_arg));
}

extern "C" __global__ void fused_gemm_max_mean_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for partial dot product
    extern __shared__ float sdata[];

    // GEMM: compute weight @ input for this (row, col)
    float sum = 0.0f;
    for (int k = tid; k < in_features; k += blockDim.x) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float result = sdata[0];
    if (bias != nullptr) {
        result += bias[col];
    }

    // Store full GEMM + bias result in shared memory for row-wise reduction
    __shared__ float row_outputs[8192]; // Max out_features = 8192
    if (col < out_features) {
        row_outputs[col] = result;
    }
    __syncthreads();

    // Only one thread per row performs max and mean reduction
    float max_val = -FLT_MAX;
    float mean_val = 0.0f;
    if (col == 0) {
        for (int i = 0; i < out_features; i++) {
            float val = row_outputs[i];
            max_val = fmaxf(max_val, val);
            mean_val += val;
        }
        mean_val /= out_features;

        // Overwrite shared memory with GELU-activated, centered values
        for (int i = 0; i < out_features; i++) {
            float centered = row_outputs[i] - mean_val;
            row_outputs[i] = gelu_activation(centered);
        }
    }
    __syncthreads();

    // Write final output: we are reducing max over dim=1, but keeping shape (batch_size, out_features)
    // However, the original model does: torch.max(x, dim=1, keepdim=True).values -> (batch_size, 1)
    // Then subtracts mean(dim=1, keepdim=True) from that (batch_size, 1) tensor
    // So we must first compute max along dim=1 -> (batch_size, 1), then use that to center the original GEMM output?

    // Wait: let's re-analyze the original forward pass:
    // x = gemm(x)  -> (B, O)
    // x = torch.max(x, dim=1, keepdim=True).values -> (B, 1)
    // x = x - x.mean(dim=1, keepdim=True) -> (B, 1) - (B, 1) = (B, 1)
    // x = gelu(x) -> (B, 1)

    // So output should be (B, 1), not (B, O)

    // Therefore, we only need to output one value per row: the GELU of (max - mean)

    if (col == 0) {
        float max_centered = max_val - mean_val;
        output[row] = gelu_activation(max_centered);
    }
}
"""

fused_cpp_source = """
torch::Tensor fused_gemm_max_mean_gelu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
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
    Optimized model with fused GEMM + max(dim=1) + mean subtraction + GELU in a single CUDA kernel.
    Output shape is (batch_size, 1) as per original model.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.max_dim = max_dim
        self.fused_op = fused_extension.fused_gemm_max_mean_gelu

    def forward(self, x):
        # The fused kernel computes GEMM, then max over dim=1 (per row), subtracts mean of that scalar (itself), then applies GELU
        # Returns (batch_size, 1)
        output = self.fused_op(x, self.linear.weight, self.linear.bias)
        return output.unsqueeze(-1)  # Ensure shape (B, 1)