import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + Scale + BatchNorm3d + AdaptiveAvgPool3d
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: performs transposed convolution, scaling, batch norm, and global avg pool
__global__ void fused_conv_transpose_scale_bn_gap_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_weight,
    const float* bn_bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_size,
    float scale_factor,
    float eps,
    int out_depth,
    int out_height,
    int out_width)
{
    int n = blockIdx.x;
    int oc = blockIdx.y;

    if (n >= batch_size || oc >= out_channels) return;

    const int pad = kernel_size / 2;
    float sum = 0.0f;

    // Compute output spatial size
    int total_output_elements = out_depth * out_height * out_width;
    float* temp_output = new float[total_output_elements];
    for (int idx = 0; idx < total_output_elements; ++idx) {
        temp_output[idx] = 0.0f;
    }

    // Transposed convolution: slide kernel and add to output feature map
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    float w = weight[oc * in_channels * kernel_size * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size * kernel_size +
                                     kd * kernel_size * kernel_size +
                                     kh * kernel_size +
                                     kw];
                    for (int id = 0; id < in_depth; ++id) {
                        for (int ih = 0; ih < in_height; ++ih) {
                            for (int iw = 0; iw < in_width; ++iw) {
                                int od = id + kd - pad;
                                int oh = ih + kh - pad;
                                int ow = iw + kw - pad;

                                if (od >= 0 && od < out_depth && oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                                    int in_idx = n * in_channels * in_depth * in_height * in_width +
                                                 ic * in_depth * in_height * in_width +
                                                 id * in_height * in_width + ih * in_width + iw;
                                    int out_idx = od * out_height * out_width + oh * out_width + ow;
                                    temp_output[out_idx] += w * input[in_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias
    for (int idx = 0; idx < total_output_elements; ++idx) {
        temp_output[idx] += bias[oc];
    }

    // Scale
    for (int idx = 0; idx < total_output_elements; ++idx) {
        temp_output[idx] *= scale_factor;
    }

    // Batch norm with global average pooling: compute mean over D, H, W
    float channel_sum = 0.0f;
    for (int idx = 0; idx < total_output_elements; ++idx) {
        channel_sum += temp_output[idx];
    }
    float spatial_avg = channel_sum / total_output_elements;

    float bn_val = (spatial_avg - running_mean[oc]) / sqrtf(running_var[oc] + eps) * bn_weight[oc] + bn_bias[oc];

    // Write final output (1 value per batch per channel)
    int out_idx = n * out_channels + oc;
    output[out_idx] = bn_val;

    delete[] temp_output;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_transpose_scale_bn_gap(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float scale_factor,
    float eps,
    int out_depth,
    int out_height,
    int out_width);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_transpose_scale_bn_gap",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_transpose_scale_bn_gap"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for ConvTranspose3d + Scale + BatchNorm3d + GlobalAvgPool3d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.eps = eps
        self.momentum = momentum

        # Define the transposed convolution layer
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        # Batch norm
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        # Calculate output shape of transposed conv
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        k = self.kernel_size
        p = k // 2
        out_depth = D + 2 * p - (k - 1)
        out_height = H + 2 * p - (k - 1)
        out_width = W + 2 * p - (k - 1)

        # Extract parameters
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        bn_weight = self.batch_norm.weight
        bn_bias = self.batch_norm.bias
        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var

        # Call fused CUDA kernel
        output = fused_op.fused_conv_transpose_scale_bn_gap(
            x,
            weight,
            bias,
            bn_weight,
            bn_bias,
            running_mean,
            running_var,
            self.scale_factor,
            self.eps,
            out_depth,
            out_height,
            out_width
        )

        # Reshape to match original output: (batch, out_channels, 1, 1, 1)
        output = output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return output