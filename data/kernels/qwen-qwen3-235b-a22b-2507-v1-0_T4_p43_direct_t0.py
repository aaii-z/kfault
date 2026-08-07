import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv3d + MaxPool3d + LogSumExp + ReLU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to calculate 3D index
__device__ int get_index(int d, int h, int w, int D, int H, int W) {
    return d * H * W + h * W + w;
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
    int kernel_size, int stride, int padding,
    // MaxPool is 2x2x2 with stride 2
    int mp_kernel, int mp_stride
) {
    int batch = blockIdx.x;
    int od = blockIdx.y;
    int oh = blockIdx.z / out_width;
    int ow = blockIdx.z % out_width;

    if (batch >= batch_size || od >= out_depth || oh >= out_height || ow >= out_width) return;

    int conv_out_d = od * mp_stride;
    int conv_out_h = oh * mp_stride;
    int conv_out_w = ow * mp_stride;

    // Buffer for conv output (before maxpool)
    float conv_buf[2][2][2];
    for (int kd = 0; kd < 2; ++kd) {
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                conv_buf[kd][kh][kw] = -FLT_MAX;
            }
        }
    }

    // Perform conv and maxpool together
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
                            int in_idx = get_index(batch, ic, id, ih, iw, batch_size, in_channels, in_depth, in_height, in_width);
                            int w_idx = get_index(oc, ic, kd, kh, kw, out_channels, in_channels, kernel_size, kernel_size, kernel_size);
                            conv_val += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        conv_val += bias[oc];

        // Max pooling over 2x2x2
        int local_d = conv_out_d % mp_stride;
        int local_h = conv_out_h % mp_stride;
        int local_w = conv_out_w % mp_stride;
        if (local_d < mp_kernel && local_h < mp_kernel && local_w < mp_kernel) {
            atomicMax((int*)&__float_as_int(conv_buf[local_d][local_h][local_w]), __float_as_int(conv_val));
        }
    }

    // Now do logsumexp over channel dim (keepdim=True), so we reduce channel dim to 1
    float sum_exp = 0.0f;
    for (int kd = 0; kd < 2; ++kd) {
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                float val = conv_buf[kd][kh][kw];
                if (val > -FLT_MAX) {
                    sum_exp += expf(val);
                }
            }
        }
    }

    float lse = logf(sum_exp);
    float relu_out = fmaxf(0.0f, lse);

    // Only one output channel
    int out_idx = get_index(batch, 0, od, oh, ow, batch_size, 1, out_depth, out_height, out_width);
    output[out_idx] = relu_out;
}
"""

cpp_source = """
torch::Tensor fused_conv_mp_lse_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width
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
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Same conv layer but we'll use its params
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        # Input shape
        batch_size, _, in_depth, in_height, in_width = x.shape

        # Output shape after conv and maxpool
        conv_out_depth = (in_depth + 2 * self.padding - self.kernel_size) // self.stride + 1
        conv_out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        conv_out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        out_depth = conv_out_depth // 2
        out_height = conv_out_height // 2
        out_width = conv_out_width // 2

        # Run fused kernel
        output = fused_op.fused_conv_mp_lse_relu(
            x,
            self.conv.weight,
            self.conv.bias,
            in_depth, in_height, in_width,
            out_depth, out_height, out_width
        )

        return output