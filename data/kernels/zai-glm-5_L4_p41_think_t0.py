import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm + GELU + ReLU
fused_bn_gelu_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bn_gelu_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int batch_size,
    int out_features,
    float eps) {

    // blockIdx.x corresponds to the batch index (0 to batch_size - 1)
    // blockIdx.y corresponds to the feature block index
    // threadIdx.x corresponds to the feature index within the block
    int batch_idx = blockIdx.x;
    int feat_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (feat_idx < out_features) {
        // Calculate the linear index for the input/output tensor
        // Assuming row-major layout: (batch, features)
        long long idx = (long long)batch_idx * out_features + feat_idx;

        // Load input value
        float val = input[idx];

        // Load BatchNorm parameters (weight, bias, mean, var)
        // These are small tensors and access pattern is broadcast-like per row
        float w = weight[feat_idx];
        float b = bias[feat_idx];
        float m = running_mean[feat_idx];
        float v = running_var[feat_idx];

        // BatchNorm inference calculation: (x - mean) / sqrt(var + eps) * weight + bias
        float inv_std = rsqrtf(v + eps);
        float x_hat = (val - m) * inv_std;
        float x_bn = x_hat * w + b;

        // GELU calculation (exact version using error function)
        // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        float x_gelu = 0.5f * x_bn * (1.0f + erff(x_bn * 0.70710678118654752440f));

        // ReLU calculation
        float x_out = fmaxf(0.0f, x_gelu);

        // Write output
        output[idx] = x_out;
    }
}

torch::Tensor fused_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {

    auto batch_size = input.size(0);
    auto out_features = input.size(1);
    auto output = torch::empty_like(input);

    // Grid configuration:
    // dim3 blocks(batch_size, (out_features + block_size - 1) / block_size)
    // This creates a 2D grid where each row handles one batch sample
    const int block_size = 256;
    dim3 blocks(batch_size, (out_features + block_size - 1) / block_size);
    dim3 threads(block_size);

    fused_bn_gelu_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        eps
    );

    return output;
}
"""

fused_bn_gelu_relu_cpp_source = (
    "torch::Tensor fused_bn_gelu_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, float eps);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bn_gelu_relu_cpp_source,
    cuda_sources=fused_bn_gelu_relu_source,
    functions=["fused_bn_gelu_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a GEMM followed by a fused BatchNorm + GELU + ReLU kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        
        # Apply fused kernel only during inference (eval mode) to ensure correctness.
        # During training, we fall back to PyTorch's autograd-enabled operations.
        if not self.training:
            x = self.fused_ops.fused_bn_gelu_relu_cuda(
                x, 
                self.batch_norm.weight, 
                self.batch_norm.bias, 
                self.batch_norm.running_mean, 
                self.batch_norm.running_var, 
                self.batch_norm.eps
            )
        else:
            x = self.batch_norm(x)
            x = F.gelu(x)
            x = F.relu(x)
            
        return x