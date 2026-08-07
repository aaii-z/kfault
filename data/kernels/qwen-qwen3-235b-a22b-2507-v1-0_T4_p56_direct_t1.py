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

    // Shared memory for tiling
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = &shared_mem[TILE_SIZE * TILE_SIZE];

    float sum_val = 0.0f;

    // Loop over tiles of input and weight
    for (int tile_idx = 0; tile_idx < (input_size + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {
        int input_col = tile_idx * TILE_SIZE + tx;
        int weight_row = bx * TILE_SIZE + ty;

        // Load input tile: each thread loads one element
        if (input_col < input_size && batch_idx < batch_size) {
            shared_input[ty * TILE_SIZE + tx] = input[batch_idx * input_size + input_col];
        } else {
            shared_input[ty * TILE_SIZE + tx] = 0.0f;
        }

        // Load weight tile: each thread loads one element
        if (weight_row < hidden_size && input_col < input_size) {
            shared_weight[ty * TILE_SIZE + tx] = weight[weight_row * input_size + input_col];
        } else {
            shared_weight[ty * TILE_SIZE + tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        if (weight_row < hidden_size) {
            float acc = 0.0f;
            for (int k = 0; k < TILE_SIZE; ++k) {
                acc += shared_input[ty * TILE_SIZE + k] * shared_weight[k * TILE_SIZE + tx];
            }
            // Accumulate in local register
            sum_val += acc;
        }

        __syncthreads();
    }

    // Add bias and apply sigmoid for each neuron (hidden unit)
    int neuron_idx = bx * TILE_SIZE + ty;
    float result = 0.0f;
    if (neuron_idx < hidden_size) {
        float pre_act = sum_val + bias[neuron_idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-fmaxf(fminf(pre_act, 88.0f), -88.0f))); // clamp to avoid overflow
        result = sigmoid_val;
    }

    // Use shared memory to reduce within block
    shared_input[ty * TILE_SIZE + tx] = result;
    __syncthreads();

    // Reduce sum within the block using shared memory
    for (int stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            shared_input[ty * TILE_SIZE + tx] += shared_input[ty * TILE_SIZE + tx + stride];
        }
        __syncthreads();
    }

    // First thread in each row writes partial sum
    if (tx == 0) {
        atomicAdd(&output[batch_idx], shared_input[ty * TILE_SIZE]);
    }
}
"""

fused_linear_sigmoid_sum_cpp_source = """
torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Wrapper that launches the kernel with appropriate grid/block dimensions
fused_linear_sigmoid_sum_source += """
torch::Tensor fused_linear_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::zeros({batch_size, 1}, input.options());

    const int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((hidden_size + TILE_SIZE - 1) / TILE_SIZE, batch_size);

    int shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    fused_linear_sigmoid_sum_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return output;
}
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_linear_sigmoid_sum",
    cpp_sources=fused_linear_sigmoid_sum_cpp_source,
    cuda_sources=fused_linear_sigmoid_sum_source,
    functions=["fused_linear_sigmoid_sum_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused linear + sigmoid + sum using custom CUDA kernel.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fused_op = fused_op.fused_linear_sigmoid_sum_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias)