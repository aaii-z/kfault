import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + max + mean subtraction + GELU
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

    float local_result = sdata[0];
    if (bias != nullptr) {
        local_result += bias[col];
    }

    // Only one thread per row handles reduction and activation
    __shared__ float row_data[8192]; // Max out_features = 8192
    if (col < out_features) {
        row_data[col] = local_result;
    }
    __syncthreads();

    if (col == 0) {
        float max_val = -FLT_MAX;
        float mean_val = 0.0f;
        for (int i = 0; i < out_features; i++) {
            float val = row_data[i];
            max_val = fmaxf(max_val, val);
            mean_val += val;
        }
        mean_val /= out_features;

        for (int i = 0; i < out_features; i++) {
            float centered = row_data[i] - mean_val;
            row_data[i] = gelu_activation(centered);
        }
    }
    __syncthreads();

    if (col < out_features) {
        output[row * out_features + col] = row_data[col];
    }
}

torch::Tensor fused_gemm_max_mean_gelu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    dim3 block_dim(block_size);
    dim3 grid_dim((out_features + block_size - 1) / block_size, batch_size);

    int shared_mem_size = block_size * sizeof(float);

    fused_gemm_max_mean_gelu_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
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
    Optimized model with fused GEMM + max + mean subtraction + GELU in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.max_dim = max_dim
        self.fused_op = fused_extension.fused_gemm_max_mean_gelu

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias)