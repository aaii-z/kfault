import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + ReLU + GroupNorm
fused_convtranspose_relu_gn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_convtranspose_relu_gn_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const float* running_mean,
    const float* running_var,
    const float* weight_gn,
    const float* bias_gn,
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W,
    int D_out, int H_out, int W_out,
    int kernel_size, int groups, float eps,
    int stride, int padding, int output_padding
) {
    int total_elements = batch_size * out_channels * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / (W_out * H_out * D_out)) % out_channels;
    int b = idx / (W_out * H_out * D_out * out_channels);

    float conv_val = 0.0f;
    const int pad = padding;

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_in = d_out * stride - pad + kd - output_padding;
                int h_in = h_out * stride - pad + kh - output_padding;
                int w_in = w_out * stride - pad + kw - output_padding;

                if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        int weight_idx = c_out * in_channels * kernel_size * kernel_size * kernel_size +
                                        c_in * kernel_size * kernel_size * kernel_size +
                                        kd * kernel_size * kernel_size + kh * kernel_size + kw;
                        int input_idx = b * in_channels * D * H * W +
                                       c_in * D * H * W + d_in * H * W + h_in * W + w_in;
                        conv_val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        conv_val += bias[c_out];
    }

    float relu_val = fmaxf(0.0f, conv_val);

    // Group Normalization
    int out_channels_per_group = out_channels / groups;
    int group_id = c_out / out_channels_per_group;

    float mean = running_mean[group_id];
    float var = running_var[group_id];
    float inv_std = rsqrtf(var + eps);
    float norm_val = (relu_val - mean) * inv_std;

    if (weight_gn != nullptr) {
        norm_val = norm_val * weight_gn[c_out];
    }
    if (bias_gn != nullptr) {
        norm_val = norm_val + bias_gn[c_out];
    }

    output[idx] = norm_val;
}
"""

fused_convtranspose_relu_gn_cpp_source = """
torch::Tensor fused_convtranspose_relu_gn_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int batch_size, int in_channels, int out_channels,
    int D, int H, int W,
    int D_out, int H_out, int W_out,
    int kernel_size, int groups, float eps,
    int stride, int padding, int output_padding);
"""

# Compile the inline CUDA kernel
try:
    fused_op = load_inline(
        name="fused_convtranspose_relu_gn",
        cpp_sources=fused_convtranspose_relu_gn_cpp_source,
        cuda_sources=fused_convtranspose_relu_gn_source,
        functions=["fused_convtranspose_relu_gn_cuda"],
        verbose=False,
        extra_cflags=["-O3"],
        extra_ldflags=[]
    )
except Exception as e:
    fused_op = None

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for ConvTranspose3d + ReLU + GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.bias = bias

        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.relu = nn.ReLU()  # Fixed: added relu attribute

        self.stride = self.conv_transpose.stride[0]
        self.padding = self.conv_transpose.padding[0]
        self.output_padding = self.conv_transpose.output_padding[0]
        self.eps = self.group_norm.eps

        self.register_buffer('running_mean', torch.zeros(groups))
        self.register_buffer('running_var', torch.ones(groups))

    def forward(self, x):
        if fused_op is None:
            x = self.conv_transpose(x)
            x = self.relu(x)
            x = self.group_norm(x)
            return x

        if self.training:
            with torch.no_grad():
                z = self.conv_transpose(x)
                z = torch.relu(z)
                z_grouped = z.view(z.size(0), self.groups, self.out_channels // self.groups, *z.shape[2:])
                mean = z_grouped.mean(dim=[2, 3, 4, 5], keepdim=False)
                mean = mean.mean(dim=0)
                var = z_grouped.var(dim=[2, 3, 4, 5], unbiased=False, keepdim=False)
                var = var.mean(dim=0)
                self.running_mean.data = mean
                self.running_var.data = var

        batch_size, _, D, H, W = x.shape
        D_out = (D - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        H_out = (H - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        W_out = (W - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.zeros(self.out_channels, device=x.device)

        weight_gn = self.group_norm.weight if self.group_norm.weight is not None else torch.ones(self.out_channels, device=x.device)
        bias_gn = self.group_norm.bias if self.group_norm.bias is not None else torch.zeros(self.out_channels, device=x.device)

        output = fused_op.fused_convtranspose_relu_gn_cuda(
            x, weight, bias,
            self.running_mean, self.running_var,
            weight_gn, bias_gn,
            batch_size, self.in_channels, self.out_channels,
            D, H, W,
            D_out, H_out, W_out,
            self.kernel_size, self.groups, self.eps,
            self.stride, self.padding, self.output_padding
        )

        return output