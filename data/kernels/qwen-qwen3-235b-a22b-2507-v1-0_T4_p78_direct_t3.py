import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + MaxPool3d (2) + MaxPool3d (3) + sum over channel dim
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_conv_transpose_maxpool_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    // Input tensor shape
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Conv transpose params
    int kernel_size, int stride, int padding,
    // Output shape after conv_transpose
    int conv_out_depth, int conv_out_height, int conv_out_width,
    // Output shape after two max pools
    int out_D, int out_H, int out_W,
    int out_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_D * out_H * out_W;
    if (idx >= total_elements) return;

    int n = idx / (out_D * out_H * out_W);
    int d = (idx % (out_D * out_H * out_W)) / (out_H * out_W);
    int h = (idx % (out_H * out_W)) / out_W;
    int w = idx % out_W;

    // Pooling after conv_transpose: MaxPool3d(kernel_size=2, stride=2)
    int pool1_d_start = d * 3;
    int pool1_h_start = h * 3;
    int pool1_w_start = w * 3;
    int pool1_depth = (conv_out_depth + 1) / 2;
    int pool1_height = (conv_out_height + 1) / 2;
    int pool1_width = (conv_out_width + 1) / 2;

    float sum_val = 0.0f;

    for (int oc = 0; oc < out_channels; oc++) {
        float max_val = -FLT_MAX;

        // MaxPool3d(kernel_size=3) over pool1 output
        for (int pd = 0; pd < 3; pd++) {
            int d1 = pool1_d_start + pd;
            if (d1 >= pool1_depth) continue;
            int conv_d_start = d1 * 2;

            for (int ph = 0; ph < 3; ph++) {
                int h1 = pool1_h_start + ph;
                if (h1 >= pool1_height) continue;
                int conv_h_start = h1 * 2;

                for (int pw = 0; pw < 3; pw++) {
                    int w1 = pool1_w_start + pw;
                    if (w1 >= pool1_width) continue;
                    int conv_w_start = w1 * 2;

                    // MaxPool3d(kernel_size=2) over conv_transpose output
                    for (int dd = 0; dd < 2; dd++) {
                        int out_d = conv_d_start + dd;
                        if (out_d >= conv_out_depth) continue;

                        for (int hh = 0; hh < 2; hh++) {
                            int out_h = conv_h_start + hh;
                            if (out_h >= conv_out_height) continue;

                            for (int ww = 0; ww < 2; ww++) {
                                int out_w = conv_w_start + ww;
                                if (out_w >= conv_out_width) continue;

                                float val = 0.0f;
                                // ConvTranspose3d computation
                                for (int ic = 0; ic < in_channels; ic++) {
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

                                                int w_idx = ((oc * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size +
                                                            kh * kernel_size + kw;
                                                int x_idx = ((n * in_channels + ic) * in_depth + in_d) * in_height * in_width +
                                                            in_h * in_width + in_w;
                                                val += input[x_idx] * weight[w_idx];
                                            }
                                        }
                                    }
                                }
                                val += bias[oc];
                                max_val = fmaxf(max_val, val);
                            }
                        }
                    }
                }
            }
        }
        sum_val += max_val;
    }

    output[idx] = sum_val;
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

# Calculate output size of conv_transpose
out_depth = (depth - 1) * stride_val - 2 * padding_val + kernel_size_val
out_height = (height - 1) * stride_val - 2 * padding_val + kernel_size_val
out_width = (width - 1) * stride_val - 2 * padding_val + kernel_size_val

# After MaxPool3d(kernel_size=2, stride=2)
pool1_depth = (out_depth + 1) // 2
pool1_height = (out_height + 1) // 2
pool1_width = (out_width + 1) // 2

# After MaxPool3d(kernel_size=3)
out_D = pool1_depth - 2
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
        return output.view(x.size(0), 1, out_D, out_H, out_W)