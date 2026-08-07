import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv3d + MaxPool3d + LogSumExp + ReLU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Helper function to calculate 4D index: (batch, channel, depth, height, width)
__device__ int get_index(int b, int c, int d, int h, int w, 
                         int batch_size, int channels, int depth, int height, int width) {
    return b * (channels * depth * height * width) + 
           c * (depth * height * width) + 
           d * (height * width) + 
           h * width + 
           w;
}

__global__ void fused_conv_maxpool_logsumexp_relu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions after conv and maxpool
    int out_channels, int out_depth, int out_height, int out_width,
    // Conv parameters
    int kernel_size, int stride, int padding
) {
    int batch = blockIdx.x;
    int od = blockIdx.y;
    int oh = blockIdx.z / out_width;
    int ow = blockIdx.z % out_width;

    if (batch >= batch_size || od >= out_depth || oh >= out_height || ow >= out_width) return;

    // Compute starting spatial location in the conv output
    int conv_out_d = od * 2;  // maxpool stride is 2
    int conv_out_h = oh * 2;
    int conv_out_w = ow * 2;

    // Buffer to hold 2x2x2 responses from conv output (before maxpool)
    float max_vals[2][2][2];
    for (int kd = 0; kd < 2; ++kd) {
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                max_vals[kd][kh][kw] = -FLT_MAX;
            }
        }
    }

    // Perform convolution for all output channels and apply max pooling
    for (int oc = 0; oc < out_channels; ++oc) {
        float conv_val = 0.0f;

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int id = conv_out_d + kd - padding;
                        int ih = conv_out_h + kh - padding;
                        int iw = conv_out_w + kw - padding;

                        if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            int in_idx = get_index(batch, ic, id, ih, iw,
                                                   batch_size, in_channels, in_depth, in_height, in_width);
                            int w_idx = get_index(oc, ic, kd, kh, kw,
                                                  out_channels, in_channels, kernel_size, kernel_size, kernel_size);
                            conv_val += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        conv_val += bias[oc];

        // Max pooling over 2x2x2 region
        int local_d = conv_out_d % 2;
        int local_h = conv_out_h % 2;
        int local_w = conv_out_w % 2;
        if (local_d < 2 && local_h < 2 && local_w < 2) {
            if (conv_val > max_vals[local_d][local_h][local_w]) {
                max_vals[local_d][local_h][local_w] = conv_val;
            }
        }
    }

    // Now compute logsumexp over the 8 values (2x2x2) in the max-pooled region
    float sum_exp = 0.0f;
    for (int kd = 0; kd < 2; ++kd) {
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                float val = max_vals[kd][kh][kw];
                if (val > -FLT_MAX) {
                    sum_exp += expf(val);
                }
            }
        }
    }

    float lse = logf(sum_exp);
    float relu_out = fmaxf(0.0f, lse);

    // Output has 1 channel (keepdim=True in logsumexp)
    int out_idx = get_index(batch, 0, od, oh, ow,
                            batch_size, 1, out_depth, out_height, out_width);
    output[out_idx] = relu_out;
}
"""

cpp_source = """
torch::Tensor fused_conv_mp_lse_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_conv_mp_lse_relu",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_mp_lse_relu"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Keep the original conv layer to use its parameters
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # MaxPool3d is fused, so we don't call it directly

    def forward(self, x):
        batch_size, _, in_depth, in_height, in_width = x.shape

        # Compute output dimensions after conv and maxpool
        conv_out_depth = (in_depth + 2 * self.padding - self.kernel_size) // self.stride + 1
        conv_out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        conv_out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        out_depth = conv_out_depth // 2
        out_height = conv_out_height // 2
        out_width = conv_out_width // 2

        # Allocate output tensor (1 channel due to keepdim=True in logsumexp)
        output = torch.empty(batch_size, 1, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

        # Launch fused kernel
        grid = (batch_size, out_depth, out_height * out_width)
        block = (1, 1, 1)  # Each block handles one output voxel

        fused_op.fused_conv_mp_lse_relu(
            x, self.conv.weight, self.conv.bias,
            batch_size, self.in_channels, in_depth, in_height, in_width,
            self.out_channels, out_depth, out_height, out_width,
            self.kernel_size, self.stride, self.padding
        )

        return output