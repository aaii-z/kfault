import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sigmoid + linear + logsumexp with high parallelism
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Efficient fused kernel with proper parallelization across batch and output dimensions
__global__ void fused_gemm_sigmoid_gemm_logsumexp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight1,
    const float* __restrict__ bias1,
    const float* __restrict__ weight2,
    const float* __restrict__ bias2,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size,
    int output_size)
{
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ float sdata[];

    float sum = 0.0f;

    // Use all threads in the block to compute the first GEMM + sigmoid
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        float acc = bias1[j];
        for (int k = 0; k < input_size; k++) {
            acc += input[batch_idx * input_size + k] * weight1[j * input_size + k];
        }
        acc = 1.0f / (1.0f + expf(-acc)); // sigmoid

        if (out_idx < output_size) {
            sum += acc * weight2[out_idx * hidden_size + j];
        }
    }

    // Add bias2 if this thread is responsible for an output
    if (out_idx < output_size) {
        sum += bias2[out_idx];
    }

    // Store exp(sum) for logsumexp, or 0 if out_idx is out of bounds
    float val = (out_idx < output_size) ? expf(sum) : 0.0f;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    // Block-wide reduction in shared memory
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < (blockDim.x * blockDim.y)) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // First thread in block writes the logsumexp result
    if (tid == 0) {
        output[batch_idx] = logf(fmaxf(sdata[0], 1e-30f)); // avoid log(0)
    }
}
"""

fused_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor bias1,
    torch::Tensor weight2,
    torch::Tensor bias2);
"""

# Compile the inline CUDA kernel
fused_extension = load_inline(
    name="fused_gemm_sigmoid_gemm_logsumexp",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_forward_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_op = fused_extension.fused_forward_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias)