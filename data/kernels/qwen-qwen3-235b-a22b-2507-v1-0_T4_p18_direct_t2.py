import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_linear_sum_max_mean_logsumexp_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    float sum_val = 0.0f;
    float max_val = -1e30f;

    // Perform GEMV (fully connected) and reduction in one pass
    for (int j = 0; j < out_features; j++) {
        float val = bias[j];
        for (int i = 0; i < in_features; i++) {
            val += input[row * in_features + i] * weight[j * in_features + i];
        }
        sum_val += val;
        max_val = fmaxf(max_val, val);
    }

    // Mean over the out_features elements
    float mean_val = sum_val / out_features;

    // First logsumexp: log(sum(exp(x))) = max_val + log(sum(exp(x - max_val)))
    float acc = 0.0f;
    for (int j = 0; j < out_features; j++) {
        float val = bias[j];
        for (int i = 0; i < in_features; i++) {
            val += input[row * in_features + i] * weight[j * in_features + i];
        }
        acc += expf(val - max_val);
    }
    float logsumexp1 = max_val + logf(acc);

    // Second logsumexp: applied on a single-element tensor -> equals the element itself
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

    auto output = torch::empty({batch_size, 1}, input.options());

    const int block_size = 1;
    dim3 grid(batch_size);

    fused_linear_sum_max_mean_logsumexp_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_linear_sum_max_mean_logsumexp(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

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
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_linear_sum_max_mean_logsumexp(x, self.linear.weight, self.linear.bias)