import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + Scale + BatchNorm3d + AdaptiveAvgPool3d
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function to calculate 3D index
__device__ int get_index(int d, int h, int w, int D, int H, int W) {
    return d * H * W + h * W + w;
}

// Fused kernel: performs transposed conv, scaling, batch norm, and global avg pool
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
    const int k_half = kernel_size / 2;
    float sum = 0.0f;

    // Perform transposed convolution (direct implementation as dilation=1, stride=1, padding=k//2)
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    float w = weight[oc * in_channels * kernel_size * kernel_size * kernel_size + ic * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw];
                    for (int od = 0; od < out_depth; ++od) {
                        for (int oh = 0; oh < out_height; ++oh) {
                            for (int ow = 0; ow < out_width; ++ow) {
                                int id = od - kd + pad;
                                int ih = oh - kh + pad;
                                int iw = ow - kw + pad;

                                float val = 0.0f;
                                if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int in_idx = n * in_channels * in_depth * in_height * in_width +
                                                 ic * in_depth * in_height * in_width +
                                                 id * in_height * in_width + ih * in_width + iw;
                                    val = input[in_idx];
                                }
                                int out_idx = od * out_height * out_width + oh * out_width + ow;
                                sum += w * val;
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias
    sum += bias[oc];

    // Scale
    sum *= scale_factor;

    // Batch norm: (sum - running_mean[oc]) / sqrt(running_var[oc] + eps) * bn_weight[oc] + bn_bias[oc]
    float bn_val = (sum - running_mean[oc]) / sqrtf(running_var[oc] + eps) * bn_weight[oc] + bn_bias[oc];

    // Global average pooling is implicit since we reduce to 1x1x1
    // We accumulate bn_val across spatial dims (but here it's already per channel, so just assign)
    // Since we are computing one output per (n, oc), we write directly
    int out_idx = n * out_channels + oc;
    output[out_idx] = bn_val;
}

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
    int out_width) {
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    int out_channels = weight.size(0);

    auto output = torch::zeros({batch_size, out_channels}, options);

    const int block_size_x = 16;
    const int block_size_y = 16;
    dim3 grid(batch_size, out_channels);
    dim3 block(1, 1);

    fused_conv_transpose_scale_bn_gap_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        weight.size(4), // kernel_size
        scale_factor,
        eps,
        out_depth,
        out_height,
        out_width
    );

    return output;
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

        # Define the transposed convolution layer (we'll access its params)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        # Batch norm
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        # No need for explicit pooling since we do it in the kernel

        # Register buffer for output spatial dimensions
        self.register_buffer('out_depth', torch.tensor(0))
        self.register_buffer('out_height', torch.tensor(0))
        self.register_buffer('out_width', torch.tensor(0))

    def forward(self, x):
        # Calculate output shape of transposed conv
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        kernel = self.kernel_size
        padding = kernel // 2
        out_depth = D + 2 * padding - (kernel - 1) - 1 + 1
        out_height = H + 2 * padding - (kernel - 1) - 1 + 1
        out_width = W + 2 * padding - (kernel - 1) - 1 + 1

        # Update buffers
        self.out_depth.fill_(out_depth)
        self.out_height.fill_(out_height)
        self.out_width.fill_(out_width)

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