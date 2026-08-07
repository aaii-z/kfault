import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused bmm, instance norm, residual add, and multiply
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bmm_instance_norm_add_mul_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
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

    // Linear projection (bmm equivalent)
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[batch_idx * in_features + k] * weight[feature_idx * in_features + k];
    }
    sum += bias[feature_idx];

    // Instance Norm: (sum - mean) / sqrt(var + eps) * gamma + beta
    // For InstanceNorm2d with (N, C, 1, 1), mean and var are computed per channel per batch
    // Here we assume running_mean and running_var are already computed (inference mode)
    float mean = running_mean[feature_idx];
    float var = running_var[feature_idx];
    float normalized = (sum - mean) / sqrtf(var + eps);
    float norm_output = gamma[feature_idx] * normalized + beta[feature_idx];

    // Residual add: norm_output + y[batch_idx][feature_idx]
    float added = norm_output + y[batch_idx * out_features + feature_idx];

    // Multiply: added * y[batch_idx][feature_idx]
    output[idx] = added * y[batch_idx * out_features + feature_idx];
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor y
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;

    fused_bmm_instance_norm_add_mul_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        1e-5f
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor y
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for bmm, instance norm, residual add, and multiply.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.fused_op = fused_op

    def forward(self, x, y):
        # Extract parameters for fused kernel
        weight = self.linear.weight
        bias = self.linear.bias
        running_mean = self.instance_norm.running_mean
        running_var = self.instance_norm.running_var
        gamma = self.instance_norm.weight
        beta = self.instance_norm.bias

        return self.fused_op.fused_op_cuda(
            x, weight, bias, running_mean, running_var, gamma, beta, y
        )