import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_scale_bn_kernel(
    const float* x,
    float scale_factor,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* out,
    int spatial_size,
    int channels,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int c = (idx / spatial_size) % channels;
        float std = sqrtf(running_var[c] + eps);
        float A = scale_factor * weight[c] / std;
        float B = bias[c] - (running_mean[c] * weight[c]) / std;
        out[idx] = x[idx] * A + B;
    }
}

torch::Tensor fused_scale_bn_cuda(
    torch::Tensor x,
    float scale_factor,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    int total_elements = x.numel();
    auto sizes = x.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    int spatial_size = depth * height * width;

    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_scale_bn_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        scale_factor,
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        out.data_ptr<float>(),
        spatial_size,
        channels,
        total_elements
    );

    return out;
}
"""

fused_scale_bn_cpp = (
    "torch::Tensor fused_scale_bn_cuda("
    "torch::Tensor x, float scale_factor, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor running_mean, torch::Tensor running_var, float eps);"
)

fused_scale_bn = load_inline(
    name="fused_scale_bn",
    cpp_sources=fused_scale_bn_cpp,
    cuda_sources=fused_scale_bn_source,
    functions=["fused_scale_bn_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--threads", "4", "-arch=sm_70", "-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fused_op = fused_scale_bn

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_scale_bn_cuda(
            x,
            self.scale_factor,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        x = self.global_avg_pool(x)
        return x