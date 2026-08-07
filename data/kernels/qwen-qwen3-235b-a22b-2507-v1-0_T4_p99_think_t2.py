import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bias_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bias_gelu_kernel(
    const float* matmul_output,
    const float* bias,
    float* output,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int j = idx % out_features;
    float x = matmul_output[idx] + bias[j];

    float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
}

torch::Tensor fused_bias_gelu_cuda(torch::Tensor matmul_output, torch::Tensor bias) {
    int batch_size = matmul_output.size(0);
    int out_features = matmul_output.size(1);

    auto output = torch::empty_like(matmul_output);

    int threads = 256;
    int blocks = (batch_size * out_features + threads - 1) / threads;

    fused_bias_gelu_kernel<<<blocks, threads>>>(
        matmul_output.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

fused_bias_gelu_cpp = "torch::Tensor fused_bias_gelu_cuda(torch::Tensor matmul_output, torch::Tensor bias);"

fused_bias_gelu = load_inline(
    name="fused_bias_gelu",
    cpp_sources=fused_bias_gelu_cpp,
    cuda_sources=fused_bias_gelu_source,
    functions=["fused_bias_gelu_cuda"],
    verbose=False,
)

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(
    const float* input,
    float* output,
    int batch_size,
    int out_features
) {
    int i = blockIdx.x;
    if (i >= batch_size) return;

    float max_val = input[i * out_features];
    for (int j = 1; j < out_features; j++) {
        float val = input[i * out_features + j];
        if (val > max_val) max_val = val;
    }

    float sum_val = 0.0f;
    for (int j = 0; j < out_features; j++) {
        float val = expf(input[i * out_features + j] - max_val);
        sum_val += val;
    }

    for (int j = 0; j < out_features; j++) {
        output[i * out_features + j] = expf(input[i * out_features + j] - max_val) / sum_val;
    }
}

torch::Tensor custom_softmax_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int out_features = input.size(1);

    auto output = torch::empty_like(input);

    softmax_kernel<<<batch_size, 1>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

softmax_cpp = "torch::Tensor custom_softmax_cuda(torch::Tensor input);"

custom_softmax = load_inline(
    name="custom_softmax",
    cpp_sources=softmax_cpp,
    cuda_sources=softmax_source,
    functions=["custom_softmax_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_bias_gelu = fused_bias_gelu
        self.custom_softmax = custom_softmax

    def forward(self, x):
        x = x.mm(self.weight.t())
        x = self.fused_bias_gelu.fused_bias_gelu_cuda(x, self.bias)
        x = self.custom_softmax.custom_softmax_cuda(x)
        return x