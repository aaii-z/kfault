import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear, instance norm, residual add, and multiply
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_linear_instance_norm_add_mul_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gamma,
    const float* beta,
    const float* y,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int batch_idx = idx / out_features;
    int feature_idx = idx % out_features;

    // Linear projection: output = input @ weight^T + bias
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[batch_idx * in_features + k] * weight[feature_idx * in_features + k];
    }
    sum += bias[feature_idx];

    // Instance Normalization: compute mean and variance over spatial dimensions (H, W)
    // Here, we are mimicking InstanceNorm2d applied on (N, C, 1, 1), so mean = 0, var = 0, but we still apply affine
    // Since the input to InstanceNorm2d is unsqueezed to (N, C, 1, 1), mean and variance are computed over H*W = 1
    // So no actual normalization is needed beyond the affine transform if we assume track_running_stats is not used in training
    // However, in inference mode (default), InstanceNorm uses running stats or just applies affine if track_running_stats=False
    // We assume affine=True and track_running_stats=False (default), so it's just: gamma * x + beta
    float norm_output = gamma[feature_idx] * sum + beta[feature_idx];

    // Residual addition: norm_output + y
    float residual = norm_output + y[batch_idx * out_features + feature_idx];

    // Element-wise multiplication: residual * y
    output[idx] = residual * y[batch_idx * out_features + feature_idx];
}
"""

fused_cpp_source = """
torch::Tensor fused_linear_instance_norm_add_mul_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor y,
    float eps
);
"""

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_instance_norm_add_mul_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum, affine=True)
        self.eps = eps
        self.fused_op = fused_op

    def forward(self, x, y):
        weight = self.linear.weight
        bias = self.linear.bias
        gamma = self.instance_norm.weight
        beta = self.instance_norm.bias

        return self.fused_op.fused_linear_instance_norm_add_mul_cuda(
            x, weight, bias, gamma, beta, y, self.eps
        )