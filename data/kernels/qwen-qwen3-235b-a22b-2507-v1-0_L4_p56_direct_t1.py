import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sigmoid + sum with high parallelism
fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Efficient GEMM + Sigmoid + Row-wise Sum in one fused kernel
__global__ void fused_linear_sigmoid_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int batch_idx = by;
    if (batch_idx >= batch_size) return;

    // Shared memory for input tile
    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];

    float sum_val = 0.0f;

    // Loop over input tiles
    for (int tile_idx = 0; tile_idx < (input_size + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {
        int input_col = tile_idx * TILE_SIZE + tx;
        int input_row = batch_idx * input_size + input_col;

        // Load input tile
        if (input_col < input_size) {
            input_tile[ty][tx] = input[input_row];
        } else {
            input_tile[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Compute partial dot product for a tile of outputs
        for (int k = 0; k < TILE_SIZE; ++k) {
            int weight_row = (ty + tx * TILE_SIZE) % hidden_size;
            int weight_col = tile_idx * TILE_SIZE + k;
            if (weight_col < input_size && (ty + tx * TILE_SIZE) < hidden_size) {
                sum_val += input_tile[ty][tx] * weight[weight_row * input_size + weight_col];
            }
        }
        __syncthreads();
    }

    // Add bias and apply sigmoid
    int neuron_idx = ty + tx * TILE_SIZE;
    if (neuron_idx < hidden_size) {
        float pre_act = sum_val + bias[neuron_idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-fmaxf(fminf(pre_act, 88.0f), -88.0f))); // clamped expf
        sum_val = sigmoid_val;
    } else {
        sum_val = 0.0f;
    }

    // Block-level sum reduction for this batch element
    __shared__ float sdata[256];
    int tid = ty * TILE_SIZE + tx;
    sdata[tid] = sum_val;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx] = sdata[0];
    }
}
"""

fused_linear_sigmoid_sum_cpp_source = """
torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

fused_op = load_inline(
    name="fused_linear_sigmoid_sum",
    cpp_sources=[fused_linear_sigmoid_sum_cpp_source],
    cuda_sources=[fused_linear_sigmoid_sum_source],
    functions=["fused_linear_sigmoid_sum_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused linear + sigmoid + sum using custom high-performance CUDA kernel.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fused_op = fused_op.fused_linear_sigmoid_sum_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias).unsqueeze(-1)