import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + MaxPool3d (kernel_size=2) + MaxPool3d (kernel_size=3) + sum over channel dim
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Utility function to calculate 3D index
__device__ int get_index(int d, int h, int w, int D, int H, int W) {
    return d * H * W + h * W + w;
}

__global__ void fused_conv_transpose_maxpool_sum_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input tensor shape
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output tensor shape (after conv_transpose)
    int out_depth, int out_height, int out_width,
    // Conv transpose params
    int kernel_size, int stride, int padding,
    // Output shape after two max pools
    int post_pool2_depth, int post_pool2_height, int post_pool2_width,
    int out_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * post_pool2_depth * post_pool2_height * post_pool2_width;
    
    if (idx >= total_elements) return;

    int n = idx / (post_pool2_depth * post_pool2_height * post_pool2_width);
    int rem = idx % (post_pool2_depth * post_pool2_height * post_pool2_width);
    int d = rem / (post_pool2_height * post_pool2_width);
    int h = (rem % (post_pool2_height * post_pool2_width)) / post_pool2_width;
    int w = rem % post_pool2_width;

    float final_sum = 0.0f;

    // Output spatial location after both pooling layers
    int pool2_d_start = d * 3;
    int pool2_h_start = h * 3;
    int pool2_w_start = w * 3;

    for (int oc = 0; oc < out_channels; oc++) {
        float max_val = -FLT_MAX;

        // Traverse 3x3 region of first maxpool output (before 2nd maxpool)
        for (int pd = 0; pd < 3; pd++) {
            for (int ph = 0; ph < 3; ph++) {
                for (int pw = 0; pw < 3; pw++) {
                    int sd = pool2_d_start + pd;
                    int sh = pool2_h_start + ph;
                    int sw = pool2_w_start + pw;

                    // Check bounds for first maxpool output (post_pool1)
                    int post_pool1_depth = (out_depth + 1) / 2;
                    int post_pool1_height = (out_height + 1) / 2;
                    int post_pool1_width = (out_width + 1) / 2;

                    if (sd >= post_pool1_depth || sh >= post_pool1_height || sw >= post_pool1_width) continue;

                    // Reverse max_pool1 (kernel=2, stride=2)
                    int pool1_d_start = sd * 2;
                    int pool1_h_start = sh * 2;
                    int pool1_w_start = sw * 2;

                    for (int ppd = 0; ppd < 2; ppd++) {
                        for (int pph = 0; pph < 2; pph++) {
                            for (int ppw = 0; ppw < 2; ppw++) {
                                int dd = pool1_d_start + ppd;
                                int hh = pool1_h_start + pph;
                                int ww = pool1_w_start + ppw;

                                if (dd >= out_depth || hh >= out_height || ww >= out_width) continue;

                                float conv_val = 0.0f;
                                for (int c = 0; c < in_channels; c++) {
                                    for (int kd = 0; kd < kernel_size; kd++) {
                                        for (int kh = 0; kh < kernel_size; kh++) {
                                            for (int kw = 0; kw < kernel_size; kw++) {
                                                int in_d = (dd - kd) / stride + padding;
                                                int in_h = (hh - kh) / stride + padding;
                                                int in_w = (ww - kw) / stride + padding;

                                                // Check if this input location contributes
                                                if ((dd - kd + padding) % stride != 0 ||
                                                    (hh - kh + padding) % stride != 0 ||
                                                    (ww - kw + padding) % stride != 0) continue;

                                                if (in_d < 0 || in_h < 0 || in_w < 0 ||
                                                    in_d >= in_depth || in_h >= in_height || in_w >= in_width) continue;

                                                int weight_idx = kd * kernel_size * kernel_size * in_channels * out_channels +
                                                                 kh * kernel_size * in_channels * out_channels +
                                                                 kw * in_channels * out_channels +
                                                                 c * out_channels + oc;

                                                conv_val += input[n * in_channels * in_depth * in_height * in_width +
                                                                  c * in_depth * in_height * in_width +
                                                                  in_d * in_height * in_width +
                                                                  in_h * in_width +
                                                                  in_w] * weight[weight_idx];
                                            }
                                        }
                                    }
                                }
                                conv_val += bias[oc];
                                if (conv_val > max_val) {
                                    max_val = conv_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        final_sum += max_val;
    }

    output[idx] = final_sum;
}

torch::Tensor fused_conv_transpose_maxpool_sum(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels, int kernel_size, int stride, int padding,
    int in_depth, int in_height, int in_width
) {
    // Calculate output dimensions
    auto out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size;
    auto out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    auto out_width = (in_width - 1) * stride - 2 * padding + kernel_size;

    auto post_pool1_depth = (out_depth + 1) / 2;
    auto post_pool1_height = (out_height + 1) / 2;
    auto post_pool1_width = (out_width + 1) / 2;

    auto post_pool2_depth = (post_pool1_depth - 3) + 1;
    auto post_pool2_height = (post_pool1_height - 3) + 1;
    auto post_pool2_width = (post_pool1_width - 3) + 1;

    auto batch_size = input.size(0);

    auto output = torch::zeros({batch_size, post_pool2_depth, post_pool2_height, post_pool2_width},
                               torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int total_elements = batch_size * post_pool2_depth * post_pool2_height * post_pool2_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_transpose_maxpool_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding,
        post_pool2_depth, post_pool2_height, post_pool2_width,
        out_channels
    );

    return output;
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

# Given parameters
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size_val = 5
stride_val = 2
padding_val = 2

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
        return output.unsqueeze(1)  # Add channel dim back as [B, 1, D, H, W]