import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused matmul + add + Swish + Tanh + GELU + Hardtanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Swish: x * sigmoid(x)
// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// We fuse: matmul -> add bias -> Swish -> Tanh -> GELU -> Hardtanh

__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654;
    const float coeff = 0.044715;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

__global__ void fused_matmul_add_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) {

    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output feature index

    if (row >= batch_size || col >= out_features) return;

    // Perform matmul: output[row][col] = sum_k input[row][k] * weight[col][k]
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }

    // Add bias
    float x = sum + bias[col];

    // Apply Swish: x * sigmoid(x)
    x = x * (1.0f / (1.0f + expf(-x)));

    // Apply Tanh
    x = tanhf(x);

    // Apply GELU
    x = gelu(x);

    // Apply Hardtanh (clamp between -1 and 1)
    x = fmaxf(-1.0f, fminf(1.0f, x));

    output[row * out_features + col] = x;
}
"""

fused_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features);
"""

fused_module = load_inline(
    name="fused_module",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom CUDA kernel that fuses:
    matmul + add bias + Swish + Tanh + GELU + Hardtanh
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        batch_size = x.size(0)
        return fused_module.fused_forward_cuda(
            x, self.weight, self.add_value,
            batch_size, self.in_features, self.out_features
        )