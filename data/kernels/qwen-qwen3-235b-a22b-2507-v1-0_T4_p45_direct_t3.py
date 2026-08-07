import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sigmoid + linear + logsumexp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

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
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;

    // Shared memory for reduction
    extern __shared__ float sdata[];

    float sum = 0.0f;

    // First GEMM + Sigmoid: each thread computes one hidden unit
    for (int j = tid; j < hidden_size; j += block_size) {
        float acc = bias1[j];
        for (int k = 0; k < input_size; k++) {
            acc += input[batch_idx * input_size + k] * weight1[j * input_size + k];
        }
        acc = 1.0f / (1.0f + expf(-acc)); // sigmoid

        // Accumulate into second layer outputs
        for (int out_j = 0; out_j < output_size; out_j++) {
            atomicAdd(&sdata[out_j], acc * weight2[out_j * hidden_size + j]);
        }
    }

    __syncthreads();

    // Add bias and compute exp(x), then reduce
    float max_val = -INFINITY;
    for (int j = threadIdx.y; j < output_size; j += blockDim.y) {
        float val = sdata[j] + bias2[j];
        sdata[j] = val;
        max_val = fmaxf(max_val, val);
    }
    __syncthreads();

    // Subtract max for numerical stability
    float sum_exp = 0.0f;
    for (int j = threadIdx.y; j < output_size; j += blockDim.y) {
        float exp_val = expf(sdata[j] - max_val);
        sdata[j] = exp_val;
        sum_exp += exp_val;
    }
    __syncthreads();

    // Final reduction of exp values
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float temp = (tid + s < block_size) ? sdata[tid + s] : 0.0f;
            sdata[tid] += temp;
        }
        __syncthreads();
    }

    // First thread writes final log(sum(exp(x)))
    if (tid == 0) {
        output[batch_idx] = max_val + logf(fmaxf(sdata[0], 1e-30f));
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor bias1,
    torch::Tensor weight2,
    torch::Tensor bias2) {
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight1.size(0);
    int output_size = weight2.size(0);

    auto output = torch::empty({batch_size}, input.options());

    const int threads_per_block_x = 32;
    const int threads_per_block_y = 8;
    const dim3 block_size(threads_per_block_x, threads_per_block_y);
    const dim3 grid_size(batch_size, 1);

    // Shared memory: output_size floats for partial sums + reduction
    int smem_size = (output_size + threads_per_block_x * threads_per_block_y) * sizeof(float);

    fused_gemm_sigmoid_gemm_logsumexp_kernel<<<grid_size, block_size, smem_size>>>(
        input.data_ptr<float>(),
        weight1.data_ptr<float>(),
        bias1.data_ptr<float>(),
        weight2.data_ptr<float>(),
        bias2.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
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