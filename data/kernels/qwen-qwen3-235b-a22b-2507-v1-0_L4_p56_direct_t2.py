import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sigmoid + sum
fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

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
    int block_size = blockDim.x;

    if (batch_idx >= batch_size) return;

    float sum_val = 0.0f;

    // Perform GEMV: y = x @ W.T + b
    for (int i = 0; i < hidden_size; i++) {
        float acc = 0.0f;
        for (int j = 0; j < input_size; j++) {
            acc += input[batch_idx * input_size + j] * weight[i * input_size + j];
        }
        acc += bias[i];

        // Apply sigmoid with clamping to avoid overflow
        acc = fmaxf(-88.0f, fminf(88.0f, acc));
        float sigmoid_val = 1.0f / (1.0f + expf(-acc));
        sum_val += sigmoid_val;
    }

    output[batch_idx] = sum_val;
}

torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight.size(0);

    auto output = torch::empty({batch_size}, input.options());

    const int block_size = 256;
    const int grid_size = batch_size;

    fused_linear_sigmoid_sum_kernel<<<grid_size, block_size>>>(
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
    Optimized model with fused linear + sigmoid + sum using custom CUDA kernel.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fused_op = fused_op.fused_linear_sigmoid_sum_cuda

    def forward(self, x):
        out = self.fused_op(x, self.linear.weight, self.linear.bias)
        return out.unsqueeze(-1)