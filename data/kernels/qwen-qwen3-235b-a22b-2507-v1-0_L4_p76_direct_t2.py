import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + Bias + ReLU
fused_gemm_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_bias_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= out_features) return;

    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    output[row * out_features + col] = fmaxf(0.0f, sum + bias[col]);
}
"""

fused_gemm_bias_relu_cpp_source = """
torch::Tensor fused_gemm_bias_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Define the PyTorch interface function
fused_gemm_bias_relu = load_inline(
    name="fused_gemm_bias_relu",
    cpp_sources=fused_gemm_bias_relu_cpp_source,
    cuda_sources=fused_gemm_bias_relu_source,
    functions=["fused_gemm_bias_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom CUDA kernel that fuses GEMM, Bias addition, and ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_gemm_bias_relu.fused_gemm_bias_relu_cuda

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias)