import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sigmoid + linear + logsumexp with high parallelism
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Efficient fused forward: Gemm1 -> Sigmoid -> Gemm2 -> LogSumExp
// Uses two kernels: one for first Gemm + Sigmoid, second for fused Gemm2 + LogSumExp

__global__ void gemm_sigmoid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight1,
    const float* __restrict__ bias1,
    float* __restrict__ hidden,
    int batch_size,
    int input_size,
    int hidden_size)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= hidden_size) return;

    float sum = bias1[col];
    const float* w = weight1 + col * input_size;
    const float* x = input + row * input_size;

    // Unroll by 4 for better performance
    int i = 0;
    for (; i <= input_size - 4; i += 4) {
        sum += x[i] * w[i] + x[i+1] * w[i+1] + x[i+2] * w[i+2] + x[i+3] * w[i+3];
    }
    for (; i < input_size; ++i) {
        sum += x[i] * w[i];
    }

    hidden[row * hidden_size + col] = 1.0f / (1.0f + expf(-sum));
}

__global__ void gemm_logsumexp_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ weight2,
    const float* __restrict__ bias2,
    float* __restrict__ output,
    int batch_size,
    int hidden_size,
    int output_size)
{
    int row = blockIdx.y;
    if (row >= batch_size) return;

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Each block handles one batch sample
    float max_val = -INFINITY;

    // First pass: compute max over output features for numerical stability
    for (int col = tid; col < output_size; col += block_size) {
        float sum = bias2[col];
        for (int k = 0; k < hidden_size; k++) {
            sum += hidden[row * hidden_size + k] * weight2[col * hidden_size + k];
        }
        if (sum > max_val) max_val = sum;
    }

    // Broadcast max_val using shared memory
    sdata[tid] = max_val;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    float global_max = sdata[0];
    __syncthreads();

    // Second pass: compute exp(x - max) and sum
    float exp_sum = 0.0f;
    for (int col = tid; col < output_size; col += block_size) {
        float sum = bias2[col];
        for (int k = 0; k < hidden_size; k++) {
            sum += hidden[row * hidden_size + k] * weight2[col * hidden_size + k];
        }
        exp_sum += expf(sum - global_max);
    }

    sdata[tid] = exp_sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = global_max + logf(sdata[0]);
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor bias1,
    torch::Tensor weight2,
    torch::Tensor bias2)
{
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight1.size(0);
    auto output_size = weight2.size(0);

    // Allocate hidden layer
    auto hidden = torch::empty({batch_size, hidden_size}, input.options());
    auto output = torch::empty({batch_size}, input.options());

    // Launch first kernel: Gemm + Sigmoid
    dim3 grid1((hidden_size + 255) / 256, batch_size);
    dim3 block1(256);
    gemm_sigmoid_kernel<<<grid1, block1>>>(
        input.data_ptr<float>(),
        weight1.data_ptr<float>(),
        bias1.data_ptr<float>(),
        hidden.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    // Launch second kernel: Gemm + LogSumExp
    dim3 grid2(1, batch_size);
    dim3 block2(256);
    size_t shared_mem_size = block2.x * sizeof(float);
    gemm_logsumexp_kernel<<<grid2, block2, shared_mem_size>>>(
        hidden.data_ptr<float>(),
        weight2.data_ptr<float>(),
        bias2.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size,
        output_size
    );

    return output;
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