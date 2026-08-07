import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + BatchNorm + GELU + ReLU
fused_gemm_bn_gelu_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to fuse GEMM, BatchNorm, GELU, and ReLU
__global__ void fused_gemm_bn_gelu_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_running_mean,
    const float* __restrict__ bn_running_var,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float eps
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= out_features || row >= batch_size) return;

    // GEMM: output[row][col] = sum_k (input[row][k] * weight[col][k]) + bias[col]
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];

    // BatchNorm: (sum - mean) / sqrt(var + eps) * bn_weight + bn_bias
    float bn_mean = bn_running_mean[col];
    float bn_var = bn_running_var[col];
    float bn_std = sqrtf(bn_var + eps);
    float bn_out = (sum - bn_mean) / bn_std * bn_weight[col] + bn_bias[col];

    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float x = bn_out;
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3); // sqrt(2/π) ~ 0.7978845608028654
    float gelu_out = x * 0.5f * (1.0f + tanhf(inner));

    // ReLU: max(0, gelu_out)
    float relu_out = fmaxf(0.0f, gelu_out);

    output[row * out_features + col] = relu_out;
}

torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    const float eps = 1e-5;

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    const int grid_x = (out_features + block_size - 1) / block_size;
    dim3 block_dim(block_size);
    dim3 grid_dim(grid_x, batch_size);

    fused_gemm_bn_gelu_relu_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps
    );

    return output;
}
"""

fused_gemm_bn_gelu_relu_cpp_source = """
torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_gemm_bn_gelu_relu",
    cpp_sources=fused_gemm_bn_gelu_relu_cpp_source,
    cuda_sources=fused_gemm_bn_gelu_relu_source,
    functions=["fused_gemm_bn_gelu_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + BatchNorm + GELU + ReLU in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.fused_op = fused_op.fused_gemm_bn_gelu_relu_cuda

    def forward(self, x):
        return self.fused_op(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var
        )