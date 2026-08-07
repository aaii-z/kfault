import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM, scaling, and batch normalization
fused_gemm_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_scale_bn_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* scale,
    const float* bn_running_mean,
    const float* bn_running_var,
    const float* bn_weight,
    const float* bn_bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float eps,
    float scale_factor)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= out_features) return;

    // GEMM: compute dot product of input[row] and weight[col]
    float gemm_result = 0.0f;
    for (int k = 0; k < in_features; k++) {
        gemm_result += input[row * in_features + k] * weight[col * in_features + k];
    }
    gemm_result += bias[col];

    // Scale
    float scaled = gemm_result * scale[col];

    // BatchNorm: (x - mean) / sqrt(var + eps) * bn_weight + bn_bias
    float bn_result = (scaled - bn_running_mean[col]) / sqrtf(bn_running_var[col] + eps) * bn_weight[col] + bn_bias[col];

    output[row * out_features + col] = bn_result;
}

torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    float eps) {
    
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    const int grid_x = (out_features + block_size - 1) / block_size;
    dim3 block_dim(block_size);
    dim3 grid_dim(grid_x, batch_size);

    fused_gemm_scale_bn_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps,
        1.0f
    );

    return output;
}
"""

fused_gemm_scale_bn_cpp_source = """
torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    float eps);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_gemm_scale_bn",
    cpp_sources=fused_gemm_scale_bn_cpp_source,
    cuda_sources=fused_gemm_scale_bn_source,
    functions=["fused_gemm_scale_bn_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused GEMM + Scale + BatchNorm in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.fused_op = fused_op.fused_gemm_scale_bn_cuda

    def forward(self, x):
        # Extract all necessary parameters for the fused kernel
        weight = self.linear.weight
        bias = self.linear.bias
        bn_running_mean = self.bn.running_mean
        bn_running_var = self.bn.running_var
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        scale = self.scale

        # Launch the fused CUDA kernel
        return self.fused_op(
            x, weight, bias, scale, bn_running_mean, bn_running_var, bn_weight, bn_bias, self.eps
        )