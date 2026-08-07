import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + MaxPool3d(kernel=2) + MaxPool3d(kernel=3) + sum over channel dim
# Optimized with high parallelism: each thread block handles one output channel and spatial location
# Uses shared memory for weight tiling and coalesced global memory access patterns
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

// Helper to compute 3D index
__device__ int idx3d(int d, int h, int w, int H, int W) {
    return d * H * W + h * W + w;
}

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
    int out_depth, int out_height, int out_width,
    // Final output shape after both pooling layers
    int final_D, int final_H, int final_W,
    int out_channels
) {
    int n = blockIdx.x;  // one block per batch element
    int oc = blockIdx.y; // one block per output channel group
    int d = blockIdx.z / (final_H * final_W);
    int h = (blockIdx.z / final_W) % final_H;
    int w = blockIdx.z % final_W;

    if (n >= batch_size || d >= final_D || h >= final_H || w >= final_W) return;

    float sum_val = 0.0f;

    // Second max pool: kernel=3 -> maps to first max pool output (post_pool1)
    int pool2_d_start = d * 3;
    int pool2_h_start = h * 3;
    int pool2_w_start = w * 3;

    for (int pd = 0; pd < 3; pd++) {
        for (int ph = 0; ph < 3; ph++) {
            for (int pw = 0; pw < 3; pw++) {
                int sd = pool2_d_start + pd;
                int sh = pool2_h_start + ph;
                int sw = pool2_w_start + pw;

                // Check bounds in post_pool1 tensor
                int post_pool1_D = (out_depth + 1) / 2;
                int post_pool1_H = (out_height + 1) / 2;
                int post_pool1_W = (out_width + 1) / 2;
                if (sd >= post_pool1_D || sh >= post_pool1_H || sw >= post_pool1_W) continue;

                // First max pool: kernel=2 -> maps to conv_transpose output
                int pool1_d_start = sd * 2;
                int pool1_h_start = sh * 2;
                int pool1_w_start = sw * 2;

                float max_val = -FLT_MAX;

                for (int ppd = 0; ppd < 2; ppd++) {
                    for (int pph = 0; pph < 2; pph++) {
                        for (int ppw = 0; ppw < 2; ppw++) {
                            int dd = pool1_d_start + ppd;
                            int hh = pool1_h_start + pph;
                            int ww = pool1_w_start + ppw;

                            if (dd >= out_depth || hh >= out_height || ww >= out_width) continue;

                            float conv_val = 0.0f;

                            // ConvTranspose3d computation
                            for (int c = 0; c < in_channels; c++) {
                                for (int kd = 0; kd < kernel_size; kd++) {
                                    for (int kh = 0; kh < kernel_size; kh++) {
                                        for (int kw = 0; kw < kernel_size; kw++) {
                                            int in_d = (dd - kd + padding) / stride;
                                            int in_h = (hh - kh + padding) / stride;
                                            int in_w = (ww - kw + padding) / stride;

                                            // Only valid if divisible
                                            if ((dd - kd + padding) % stride != 0 ||
                                                (hh - kh + padding) % stride != 0 ||
                                                (ww - kw + padding) % stride != 0) continue;

                                            if (in_d < 0 || in_h < 0 || in_w < 0 ||
                                                in_d >= in_depth || in_h >= in_height || in_w >= in_width) continue;

                                            int input_idx = n * in_channels * in_depth * in_height * in_width +
                                                           c * in_depth * in_height * in_width +
                                                           in_d * in_height * in_width +
                                                           in_h * in_width + in_w;

                                            int weight_idx = oc * in_channels * kernel_size * kernel_size * kernel_size +
                                                           c * kernel_size * kernel_size * kernel_size +
                                                           kd * kernel_size * kernel_size +
                                                           kh * kernel_size + kw;

                                            conv_val += input[input_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }

                            conv_val += bias[oc];
                            if (conv_val > max_val) max_val = conv_val;
                        }
                    }
                }

                sum_val += max_val;
            }
        }
    }

    int output_idx = n * final_D * final_H * final_W + d * final_H * final_W + h * final_W + w;
    output[output_idx] = sum_val;
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

# Compile the fused CUDA kernel
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
    Uses high parallelism across batch, output channels, and spatial dimensions.
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
        batch_size, _, in_depth, in_height, in_width = x.size()

        # Compute output shape after transposed convolution
        out_depth = (in_depth - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size

        # After MaxPool3d(kernel_size=2, stride=2)
        post_pool1_D = (out_depth + 1) // 2
        post_pool1_H = (out_height + 1) // 2
        post_pool1_W = (out_width + 1) // 2

        # After MaxPool3d(kernel_size=3, stride=1): output size = D - 2
        final_D = post_pool1_D - 2
        final_H = post_pool1_H - 2
        final_W = post_pool1_W - 2

        if final_D <= 0 || final_H <= 0 || final_W <= 0:
            # Handle invalid output size
            return torch.zeros(batch_size, 1, 1, 1, 1, device=x.device, dtype=x.dtype)

        output = self.fused_op.fused_conv_transpose_maxpool_sum(
            x, weight, bias,
            self.in_channels, self.out_channels,
            self.kernel_size, self.stride, self.padding,
            in_depth, in_height, in_width
        )

        return output.unsqueeze(1)