import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sigmoid + sum with high parallelism
fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 16

// Efficient fused GEMM + Sigmoid + Row-wise Sum kernel using tiling and coalesced access
__global__ void fused_linear_sigmoid_sum_kernel(
    const float* __restrict__ input,      // [batch_size, input_size]
    const float* __restrict__ weight,     // [hidden_size, input_size]
    const float* __restrict__ bias,       // [hidden_size]
    float* __restrict__ output,           // [batch_size]
    int batch_size,
    int input_size,
    int hidden_size
) {
    int bx = blockIdx.x; // batch index
    int by = blockIdx.y; // hidden tile start index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int hidden_tile_start = by * TILE_SIZE;
    int hidden_tile_end = min(hidden_tile_start + TILE_SIZE, hidden_size);

    // Shared memory for input tile (batch x input)
    extern __shared__ float shared_input[];
    float* shared_weight = &shared_input[TILE_SIZE * input_size];

    // Load input vector for this batch into shared memory (coalesced)
    for (int i = tx * TILE_SIZE + ty; i < input_size; i += TILE_SIZE * blockDim.x) {
        shared_input[i] = input[bx * input_size + i];
    }
    __syncthreads();

    float local_sum = 0.0f;

    // Loop over tiles of hidden units
    for (int h = hidden_tile_start; h < hidden_tile_end; h += TILE_SIZE) {
        // Load weight tile: TILE_SIZE x input_size
        for (int i = 0; i < input_size; i++) {
            int widx = (h + ty) * input_size + i;
            if (h + ty < hidden_tile_end) {
                shared_weight[ty * input_size + i] = weight[widx];
            }
        }
        __syncthreads();

        // Each thread computes one hidden unit in this tile
        int local_h = h + ty;
        if (local_h < hidden_size) {
            float acc = 0.0f;
            for (int i = 0; i < input_size; i++) {
                acc += shared_input[i] * shared_weight[ty * input_size + i];
            }
            acc += bias[local_h];

            // Clamp to avoid exp overflow
            acc = fmaxf(-88.0f, fminf(88.0f, acc));
            float sigmoid_val = 1.0f / (1.0f + expf(-acc));
            local_sum += sigmoid_val;
        }
        __syncthreads();
    }

    // Reduce local_sum across threads in the block (blockDim.x * TILE_SIZE threads)
    __shared__ float shared_sums[32 * TILE_SIZE]; // max blockDim.x=32, TILE_SIZE=16
    int tid = ty * blockDim.x + tx;
    shared_sums[tid] = local_sum;
    __syncthreads();

    // Unrolled reduction for blockDim.x = 16 or 32
    if (blockDim.x * TILE_SIZE >= 64) {
        if (tid < 32) shared_sums[tid] += shared_sums[tid + 32];
        __syncthreads();
    }
    if (blockDim.x * TILE_SIZE >= 32) {
        if (tid < 16) shared_sums[tid] += shared_sums[tid + 16];
        __syncthreads();
    }
    if (tid < 8)  shared_sums[tid] += shared_sums[tid + 8];
    __syncthreads();
    if (tid < 4)  shared_sums[tid] += shared_sums[tid + 4];
    __syncthreads();
    if (tid < 2)  shared_sums[tid] += shared_sums[tid + 2];
    __syncthreads();
    if (tid < 1)  shared_sums[tid] += shared_sums[tid + 1];
    __syncthreads();

    if (tid == 0) {
        output[bx] = shared_sums[0];
    }
}
"""

fused_linear_sigmoid_sum_cpp_source = """
torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the fused operator
fused_op = load_inline(
    name="fused_linear_sigmoid_sum",
    cpp_sources=[fused_linear_sigmoid_sum_cpp_source],
    cuda_sources=[fused_linear_sigmoid_sum_source],
    functions=["fused_linear_sigmoid_sum_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-maxrregcount=64"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused linear + sigmoid + sum using custom CUDA kernel.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fused_op = fused_op.fused_linear_sigmoid_sum_cuda
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        # Ensure contiguous input
        if not x.is_contiguous():
            x = x.contiguous()
        out = self.fused_op(x, self.linear.weight, self.linear.bias)
        return out.unsqueeze(-1)