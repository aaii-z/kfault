import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Gemm, multiply, and LeakyReLU
fused_gemm_mul_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_mul_leakyrelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float multiplier,
    float negative_slope,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= out_features) return;

    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];
    sum *= multiplier;
    output[row * out_features + col] = (sum > 0) ? sum : negative_slope * sum;
}
"""

fused_gemm_mul_leakyrelu_cpp_source = """
torch::Tensor fused_gemm_mul_leakyrelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float multiplier,
    float negative_slope);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_gemm_mul_leakyrelu",
    cpp_sources=fused_gemm_mul_leakyrelu_cpp_source,
    cuda_sources=fused_gemm_mul_leakyrelu_source,
    functions=["fused_gemm_mul_leakyrelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Gemm, multiply, and LeakyReLU in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_op = fused_op.fused_gemm_mul_leakyrelu_cuda

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias, self.multiplier, self.negative_slope)