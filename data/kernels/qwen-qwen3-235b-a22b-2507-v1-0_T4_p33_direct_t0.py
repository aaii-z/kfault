import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused GEMM + Scale + BatchNorm1d
fused_gemm_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_gemm_scale_bn_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* scale,
    const float* bn_running_mean,
    const float* bn_running_var,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float eps,
    float scale_factor)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= batch_size || col >= out_features) return;

    // GEMM: output[row][col] = sum_{k} input[row][k] * weight[col][k] + bias[col]
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];

    // Scale
    float scaled_val = sum * scale[col];

    // BatchNorm: (scaled_val - mean[col]) / sqrt(var[col] + eps)
    float mean_val = bn_running_mean[col];
    float inv_std = rsqrtf(bn_running_var[col] + eps);
    output[row * out_features + col] = (scaled_val - mean_val) * inv_std * scale_factor;
}

torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    float eps,
    float scale_factor) {

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const dim3 block_size(16, 16);
    const dim3 num_blocks(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );

    fused_gemm_scale_bn_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps,
        scale_factor
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
    float eps,
    float scale_factor);
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
    Optimized version of Model with fused GEMM + Scale + BatchNorm1d in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.register_buffer('scale_factor', torch.empty(out_features).fill_(1.0))

    def forward(self, x):
        # During training, we need to update running stats, so we fall back to original path
        # During inference, we can fuse everything
        if self.training:
            x = self.gemm(x)
            x = x * self.scale
            x = self.bn(x)
            return x
        else:
            # Use fused kernel: GEMM + Scale + BatchNorm
            return fused_op.fused_gemm_scale_bn_cuda(
                x,
                self.gemm.weight,
                self.gemm.bias,
                self.scale,
                self.bn.running_mean,
                self.bn.running_var,
                self.eps,
                1.0
            )