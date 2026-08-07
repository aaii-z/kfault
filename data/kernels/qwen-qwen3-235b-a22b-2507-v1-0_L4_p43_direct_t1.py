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

__global__ void fused_conv_maxpool_logsumexp_relu_kernel(
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
    int conv_d_start = od * 2;  // maxpool stride is 2
    int conv_h_start = oh * 2;
    int conv_w_start = ow * 2;

    // Buffer to store 2x2x2 maxpooled values per output channel
    float max_vals[2][2][2];
    for (int kd = 0; kd < 2; ++kd) {
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                max_vals[kd][kh][kw] = -FLT_MAX;
            }
        }
    }

    // Perform convolution over all input channels and update max_vals
    for (int oc = 0; oc < out_channels; ++oc) {
        float conv_val = 0.0f;

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int id = conv_d_start + kd - padding;
                        int ih = conv_h_start + kh - padding;
                        int iw = conv_w_start + kw - padding;

                        if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            int in_idx = get_index(batch, ic, id, ih, iw, 
                                                   batch_size, in_channels, in_depth, in_height, in_width);
                            int w_idx = ((oc * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + 
                                        kh * kernel_size + kw;
                            conv_val += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        conv_val += bias[oc];

        // Update the 2x2x2 maxpool buffer
        int local_d = conv_d_start % 2;
        int local_h = conv_h_start % 2;
        int local_w = conv_w_start % 2;
        if (local_d == 0 && local_h == 0 && local_w == 0) {
            // Only update at top-left of 2x2x2
            if (conv_val > max_vals[0][0][0]) {
                max_vals[0][0][0] = conv_val;
            }
        } else {
            int ld = local_d ? 1 : 0;
            int lh = local_h ? 1 : 0;
            int lw = local_w ? 1 : 0;
            if (conv_val > max_vals[ld][lh][lw]) {
                max_vals[ld][lh][lw] = conv_val;
            }
        }
    }

    // Now compute logsumexp over the 8 values (2x2x2) from maxpool
    float max_val = -FLT_MAX;
    for (int kd = 0; kd < 2; ++kd) {
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                if (max_vals[kd][kh][kw] > max_val) {
                    max_val = max_vals[kd][kh][kw];
                }
            }
        }
    }

    float sum_exp = 0.0f;
    for (int kd = 0; kd < 2; ++kd) {
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                if (max_vals[kd][kh][kw] > -FLT_MAX) {
                    sum_exp += expf(max_vals[kd][kh][kw] - max_val);
                }
            }
        }
    }

    float lse = max_val + logf(sum_exp);
    float relu_out = fmaxf(0.0f, lse);

    // Output has 1 channel (keepdim=True)
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
    int kernel_size, int stride, int padding
);
"""

# Compile the CUDA kernel
fused_op = load_inline(
    name="fused_conv_mp_lse_relu",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_mp_lse_relu"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Use standard Conv3d for weight/bias initialization
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        batch_size, _, in_depth, in_height, in_width = x.shape

        # Compute output spatial dimensions after conv
        conv_out_depth = (in_depth + 2 * self.padding - self.kernel_size) // self.stride + 1
        conv_out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        conv_out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # After maxpooling with 2x2x2
        out_depth = conv_out_depth // 2
        out_height = conv_out_height // 2
        out_width = conv_out_width // 2

        # Allocate output tensor (1 channel due to keepdim=True in logsumexp)
        output = torch.empty(batch_size, 1, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

        # Launch fused kernel
        weight = self.conv.weight
        bias = self.conv.bias

        # Ensure contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        output = output.contiguous()

        fused_op.fused_conv_mp_lse_relu(
            x, weight, bias,
            batch_size, self.in_channels, in_depth, in_height, in_width,
            self.out_channels, out_depth, out_height, out_width,
            self.kernel_size, self.stride, self.padding
        )

        return output