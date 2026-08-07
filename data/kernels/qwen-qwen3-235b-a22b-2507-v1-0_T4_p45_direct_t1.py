import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sigmoid + linear + logsumexp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_gemm_sigmoid_gemm_logsumexp_kernel(
    const float* input,
    const float* weight1,
    const float* bias1,
    const float* weight2,
    const float* bias2,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size,
    int output_size)
{
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    // Each block handles one batch sample; threadIdx.x is the thread index
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory for logsumexp reduction
    extern __shared__ float sdata[];

    float sum = 0.0f;

    // First GEMM: input * weight1 + bias1, followed by sigmoid
    for (int j = 0; j < hidden_size; j++) {
        float acc = bias1[j];
        for (int k = 0; k < input_size; k++) {
            acc += input[batch_idx * input_size + k] * weight1[j * input_size + k];
        }
        acc = 1.0f / (1.0f + expf(-acc)); // sigmoid

        // Second GEMM: accumulate into output space
        for (int out_dim = 0; out_dim < output_size; out_dim++) {
            sum += acc * weight2[out_dim * hidden_size + j];
        }
    }

    // Add bias2
    sum += bias2[0];

    // Store exp(sum) for logsumexp (only one value per batch)
    float exp_val = expf(sum);
    sdata[tid] = exp_val;
    __syncthreads();

    // Reduce within block to compute sum of exps
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write final logsumexp result
    if (tid == 0) {
        output[batch_idx] = logf(sdata[0]);
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

    auto output = torch::empty({batch_size}, input.options());

    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    size_t shared_mem_size = block_size * sizeof(float);

    fused_gemm_sigmoid_gemm_logsumexp_kernel<<<grid, block, shared_mem_size>>>(
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
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_op = fused_extension.fused_forward_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias)