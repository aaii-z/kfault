import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sigmoid + sum
fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void fused_linear_sigmoid_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    float sum_val = 0.0f;

    // Shared memory for reduction
    __shared__ float shared_sum[BLOCK_SIZE];

    // Each thread computes a chunk of the hidden units
    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int k = 0; k < input_size; ++k) {
            acc += input[batch_idx * input_size + k] * weight[i * input_size + k];
        }
        acc += bias[i];
        // Sigmoid with clamping to avoid exp overflow
        float sigmoid_val = 1.0f / (1.0f + expf(-fmaxf(fminf(acc, 88.0f), -88.0f)));
        thread_sum += sigmoid_val;
    }

    shared_sum[tid] = thread_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // First thread in block writes the result
    if (tid == 0) {
        output[batch_idx] = shared_sum[0];
    }
}
"""

fused_linear_sigmoid_sum_cpp_source = """
torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

fused_linear_sigmoid_sum_source += """
torch::Tensor fused_linear_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::zeros({batch_size}, input.options().dtype(torch::kFloat));

    dim3 block(BLOCK_SIZE);
    dim3 grid(batch_size);

    fused_linear_sigmoid_sum_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    // Reshape to (batch_size, 1)
    return output.unsqueeze(1);
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