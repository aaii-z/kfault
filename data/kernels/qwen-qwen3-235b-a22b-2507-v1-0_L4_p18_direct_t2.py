import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sum + max + mean + logsumexp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void fused_linear_sum_max_mean_logsumexp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    float sum_val = 0.0f;
    float max_val = -1e30f;

    // Compute linear transformation: output[j] = sum_i(input[i] * weight[j][i]) + bias[j]
    // Then reduce across out_features for sum, max, mean, and logsumexp
    for (int j = 0; j < out_features; j++) {
        float linear_val = bias[j];
        for (int i = 0; i < in_features; i++) {
            linear_val += input[row * in_features + i] * weight[j * in_features + i];
        }
        sum_val += linear_val;
        max_val = fmaxf(max_val, linear_val);
    }

    float mean_val = sum_val / out_features;

    // First logsumexp: log(sum(exp(x))) = max_val + log(sum(exp(x - max_val)))
    float acc = 0.0f;
    for (int j = 0; j < out_features; j++) {
        float linear_val = bias[j];
        for (int i = 0; i < in_features; i++) {
            linear_val += input[row * in_features + i] * weight[j * in_features + i];
        }
        acc += expf(linear_val - max_val);
    }
    float logsumexp1 = max_val + logf(acc);

    // Second logsumexp: applied on a single value -> equals the value itself
    float logsumexp2 = logsumexp1;

    output[row] = logsumexp2;
}

torch::Tensor fused_linear_sum_max_mean_logsumexp(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = bias.size(0);

    auto output = torch::empty(batch_size, input.options());

    const int block_size = BLOCK_SIZE;
    const int num_blocks = batch_size;

    fused_linear_sum_max_mean_logsumexp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output.unsqueeze(-1);
}
"""

fused_cpp_source = """
torch::Tensor fused_linear_sum_max_mean_logsumexp(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_linear_sum_max_mean_logsumexp",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_sum_max_mean_logsumexp"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel replacing:
        - Linear (matmul + bias)
        - sum(dim=1)
        - max(dim=1)
        - mean(dim=1)
        - logsumexp(dim=1)
        - logsumexp(dim=1)
    All fused into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_linear_sum_max_mean_logsumexp(x, self.linear.weight, self.linear.bias)