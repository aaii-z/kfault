import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sigmoid + linear + logsumexp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_gemm_sigmoid_gemm_logsumexp(
    const float* input,
    const float* weight1,
    const float* bias1,
    const float* weight2,
    const float* bias2,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size,
    int output_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Shared memory for partial sums in logsumexp
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

    // Add bias2 for the final output neuron
    sum += bias2[0]; // assuming bias2 is broadcasted; we take first element for logsumexp

    // Perform logsumexp: log(sum(exp(x_i))) -> we only have one scalar per batch here due to reduction
    float exp_val = expf(sum);
    
    // Store exp(x) in shared memory for reduction
    sdata[threadIdx.x] = exp_val;
    __syncthreads();

    // Simple reduction for sum of exps
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((threadIdx.x % (2 * s)) == 0) {
            if (threadIdx.x + s < blockDim.x) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // Finalize logsumexp: log(sum_exp)
    if (threadIdx.x == 0) {
        float logsumexp = logf(sdata[0]);
        output[batch_idx] = logsumexp;
    }
}
"""

fused_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor bias1,
    torch::Tensor weight2,
    torch::Tensor bias2
);
"""

# Compile the inline CUDA kernel
fused_extension = load_inline(
    name="fused_gemm_sigmoid_gemm_logsumexp",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_forward_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_op = fused_extension.fused_forward_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias)