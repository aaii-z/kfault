import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, Mul, and LeakyReLU
fused_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void fused_gemm_leaky_relu_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ out,
    int batch_size, int in_features, int out_features, 
    float multiplier, float negative_slope) {
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    if (row >= batch_size || col >= out_features) return;

    float value = 0.0f;

    // Shared memory with padding to avoid bank conflicts
    __shared__ float x_tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float w_tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    int num_tiles = (in_features + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load x tile
        int x_col = t * BLOCK_SIZE + tx;
        if (row < batch_size && x_col < in_features) {
            x_tile[ty][tx] = x[row * in_features + x_col];
        } else {
            x_tile[ty][tx] = 0.0f;
        }

        // Load weight tile
        // weight is (out_features, in_features)
        // We need weight[col, t*BS + ty]
        int w_col = t * BLOCK_SIZE + ty;
        if (col < out_features && w_col < in_features) {
            w_tile[tx][ty] = weight[col * in_features + w_col];
        } else {
            w_tile[tx][ty] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            value += x_tile[ty][k] * w_tile[tx][k];
        }

        __syncthreads();
    }

    value += bias[col];
    value *= multiplier;
    value = (value > 0.0f) ? value : value * negative_slope;

    out[row * out_features + col] = value;
}

torch::Tensor fused_gemm_leaky_relu_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
    float multiplier, float negative_slope) {
    
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0); // weight is (out, in)
    
    auto out = torch::zeros({batch_size, out_features}, x.options());
    
    dim3 blocks((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    
    fused_gemm_leaky_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), batch_size, in_features, out_features, 
        multiplier, negative_slope);
        
    return out;
}
"""

fused_gemm_cpp_source = (
    "torch::Tensor fused_gemm_leaky_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float multiplier, float negative_slope);"
)

# Compile the inline CUDA code
fused_gemm_op = load_inline(
    name="fused_gemm_op",
    cpp_sources=fused_gemm_cpp_source,
    cuda_sources=fused_gemm_source,
    functions=["fused_gemm_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a Gemm, multiplies the result, and applies LeakyReLU
    using a custom fused CUDA kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.fused_op = fused_gemm_op

    def forward(self, x):
        # Use the custom fused kernel
        return self.fused_op.fused_gemm_leaky_relu_cuda(
            x, self.gemm.weight, self.gemm.bias, self.multiplier, self.negative_slope
        )