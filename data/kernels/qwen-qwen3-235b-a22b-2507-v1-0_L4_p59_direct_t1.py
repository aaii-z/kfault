import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul + swish + scale with optimized GEMM-like tiling
fused_matmul_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16

__global__ void fused_matmul_swish_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float scaling_factor,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int tile = 0; tile < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int k = tile * TILE_SIZE + threadIdx.x;
        input_tile[threadIdx.y][threadIdx.x] = (row < batch_size && k < in_features)
            ? input[row * in_features + k] : 0.0f;

        k = tile * TILE_SIZE + threadIdx.y;
        weight_tile[threadIdx.y][threadIdx.x] = (k < in_features && col < out_features)
            ? weight[col * in_features + k] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += input_tile[threadIdx.y][i] * weight_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < batch_size && col < out_features) {
        sum += bias[col];
        float swish = sum * (1.0f / (1.0f + expf(-sum)));
        output[row * out_features + col] = swish * scaling_factor;
    }
}
"""

fused_matmul_swish_scale_cpp_source = """
torch::Tensor fused_matmul_swish_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor
);
"""

fused_op = load_inline(
    name="fused_matmul_swish_scale",
    cpp_sources=fused_matmul_swish_scale_cpp_source,
    cuda_sources=fused_matmul_swish_scale_source,
    functions=["fused_matmul_swish_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused matmul + swish activation + scaling in a single CUDA kernel.
    Uses tiled matrix multiplication for better memory coalescing and parallelism.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_matmul_swish_scale_cuda(
            x, self.linear.weight, self.linear.bias, self.scaling_factor
        )