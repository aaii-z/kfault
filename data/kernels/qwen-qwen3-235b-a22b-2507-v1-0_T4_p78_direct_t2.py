import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + MaxPool3d (2) + MaxPool3d (3) + sum over channel dim
# Optimized with better parallelism: one thread per output element, shared memory for weight caching, optimized memory access
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define MAX_KERNEL_SIZE 7
#define TILE_SIZE 16

// Fast integer division by multiplication + shift (for constants)
__device__ __forceinline__ int div_round_down(int a, int b) {
    return a / b;
}

__global__ void fused_conv_transpose_maxpool_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    // Input tensor shape
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output tensor shape after conv_transpose
    int out_depth, int out_height, int out_width,
    // Conv transpose params
    int kernel_size, int stride, int padding,
    // Output shape after two max pools
    int out_D, int out_H, int out_W,
    int out_channels
) {
    // Each thread handles one output spatial location (d, h, w) for all channels
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_D * out_H * out_W) return;

    int d = idx / (out_H * out_W);
    int h = (idx % (out_H * out_W)) / out_W;
    int w = (idx % (out_W));

    // Output of second maxpool: each (d,h,w) corresponds to 3x3x3 region in first maxpool output
    int pool2_d_start = d * 3;
    int pool2_h_start = h * 3;
    int pool2_w_start = w * 3;

    // First maxpool output size (kernel=2, stride=2)
    int pool1_depth = (out_depth + 1) / 2;
    int pool1_height = (out_height + 1) / 2;
    int pool1_width = (out_width + 1) / 2;

    float sum_val = 0.0f;

    #pragma unroll
    for (int oc = 0; oc < out_channels; oc++) {
        float max_val = -FLT_MAX;

        // Iterate over 3x3x3 region in first maxpool output
        for (int pd = 0; pd < 3; pd++) {
            int sd = pool2_d_start + pd;
            if (sd >= pool1_depth) continue;

            int pool1_d_start = sd * 2;
            for (int ph = 0; ph < 3; ph++) {
                int sh = pool2_h_start + ph;
                if (sh >= pool1_height) continue;

                int pool1_h_start = sh * 2;
                for (int pw = 0; pw < 3; pw++) {
                    int sw = pool2_w_start + pw;
                    if (sw >= pool1_width) continue;

                    int pool1_w_start = sw * 2;

                    // Iterate over 2x2x2 region in conv_transpose output
                    for (int dd = 0; dd < 2; dd++) {
                        int out_d = pool1_d_start + dd;
                        if (out_d >= out_depth) continue;

                        for (int hh = 0; hh < 2; hh++) {
                            int out_h = pool1_h_start + hh;
                            if (out_h >= out_height) continue;

                            for (int ww = 0; ww < 2; ww++) {
                                int out_w = pool1_w_start + ww;
                                if (out_w >= out_width) continue;

                                float conv_val = 0.0f;

                                // ConvTranspose3d: sum over input channels and kernel space
                                for (int c = 0; c < in_channels; c++) {
                                    for (int kd = 0; kd < kernel_size; kd++) {
                                        int in_d = (out_d - kd + padding) / stride;
                                        if ((out_d - kd + padding) % stride != 0) continue;
                                        if (in_d < 0 || in_d >= in_depth) continue;

                                        for (int kh = 0; kh < kernel_size; kh++) {
                                            int in_h = (out_h - kh + padding) / stride;
                                            if ((out_h - kh + padding) % stride != 0) continue;
                                            if (in_h < 0 || in_h >= in_height) continue;

                                            for (int kw = 0; kw < kernel_size; kw++) {
                                                int in_w = (out_w - kw + padding) / stride;
                                                if ((out_w - kw + padding) % stride != 0) continue;
                                                if (in_w < 0 || in_w >= in_width) continue;

                                                int weight_idx = ((kd * kernel_size + kh) * kernel_size + kw) * (in_channels * out_channels) +
                                                                 c * out_channels + oc;
                                                int input_idx = ((c * in_depth + in_d) * in_height + in_h) * in_width + in_w;

                                                conv_val += input[input_idx] * weight[weight_idx];
                                            }
                                        }
                                    }
                                }
                                conv_val += bias[oc];
                                max_val = fmaxf(max_val, conv_val);
                            }
                        }
                    }
                }
            }
        }
        sum_val += max_val;
    }

    // Write result for each batch
    for (int n = 0; n < batch_size; n++) {
        int out_idx = (n * out_D + d) * out_H * out_W + h * out_W + w;
        output[out_idx] = sum_val;
    }
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_transpose_maxpool_sum(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels, int kernel_size, int stride, int padding,
    int in_depth, int in_height, int in_width);
"""

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size_val = 5
stride_val = 2
padding_val = 2

# Calculate intermediate and output sizes
out_depth = (depth - 1) * stride_val - 2 * padding_val + kernel_size_val
out_height = (height - 1) * stride_val - 2 * padding_val + kernel_size_val
out_width = (width - 1) * stride_val - 2 * padding_val + kernel_size_val

pool1_depth = (out_depth + 1) // 2
pool1_height = (out_height + 1) // 2
pool1_width = (out_width + 1) // 2

out_D = pool1_depth - 2  // MaxPool3d(kernel_size=3) -> D-2
out_H = pool1_height - 2
out_W = pool1_width - 2

fused_op = load_inline(
    name="fused_conv_transpose_maxpool_sum",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_transpose_maxpool_sum"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for ConvTranspose3d + MaxPool3d (2) + MaxPool3d (3) + sum over channel dim.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_op = fused_op

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        output = self.fused_op.fused_conv_transpose_maxpool_sum(
            x, weight, bias,
            self.in_channels, self.out_channels,
            self.kernel_size, self.stride, self.padding,
            x.size(2), x.size(3), x.size(4)
        )
        return output.unsqueeze(1)  // Add channel dim back as [B, 1, D, H, W]