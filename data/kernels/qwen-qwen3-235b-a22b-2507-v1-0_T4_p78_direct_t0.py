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
    int post_pool1_depth, int post_pool1_height, int post_pool1_width,
    int post_pool2_depth, int post_pool2_height, int post_pool2_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * post_pool2_depth * post_pool2_height * post_pool2_width;
    
    if (idx >= total_elements) return;

    int n = idx / (post_pool2_depth * post_pool2_height * post_pool2_width);
    int rem = idx % (post_pool2_depth * post_pool2_height * post_pool2_width);
    int d = rem / (post_pool2_height * post_pool2_width);
    int h = (rem % (post_pool2_height * post_pool2_width)) / post_pool2_width;
    int w = rem % post_pool2_width;

    // Reverse map through max_pool2 (kernel=3, stride=1 for first two steps, then downsample)
    int pool2_d_start = d * 3;
    int pool2_h_start = h * 3;
    int pool2_w_start = w * 3;

    float max_val = -FLT_MAX;

    // Second max pool (kernel=3)
    for (int pd = 0; pd < 3; pd++) {
        for (int ph = 0; ph < 3; ph++) {
            for (int pw = 0; pw < 3; pw++) {
                int sd = pool2_d_start + pd;
                int sh = pool2_h_start + ph;
                int sw = pool2_w_start + pw;

                if (sd >= post_pool1_depth || sh >= post_pool1_height || sw >= post_pool1_width) continue;

                // Reverse max_pool1 (kernel=2)
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
                                            int in_d = dd - padding + kd;
                                            int in_h = hh - padding + kh;
                                            int in_w = ww - padding + kw;

                                            if (in_d < 0 || in_h < 0 || in_w < 0 || 
                                                in_d >= in_depth || in_h >= in_height || in_w >= in_width) continue;

                                            int weight_idx = get_index(kd, kh, kw, kernel_size, kernel_size, kernel_size) + 
                                                             c * kernel_size * kernel_size * kernel_size +
                                                             0 * in_channels * kernel_size * kernel_size * kernel_size; // out_ch=0 offset
                                            conv_val += input[n * in_channels * in_depth * in_height * in_width +
                                                              c * in_depth * in_height * in_width +
                                                              in_d * in_height * in_width +
                                                              in_h * in_width +
                                                              in_w] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                            conv_val += bias[0]; // Add bias for first output channel

                            if (conv_val > max_val) {
                                max_val = conv_val;
                            }
                        }
                    }
                }
            }
        }
    }

    // Now sum over all input channels (but we already did in the loop above via bias and weights)
    // However, note: original model does sum over output channels after pooling
    // But in our case, conv_transpose produces 'out_channels' outputs, then we sum over dim=1
    // So we need to compute all output channels and sum them

    float final_sum = 0.0f;
    for (int oc = 0; oc < 64; oc++) { // out_channels = 64
        float local_max = -FLT_MAX;
        for (int pd = 0; pd < 3; pd++) {
            for (int ph = 0; ph < 3; ph++) {
                for (int pw = 0; pw < 3; pw++) {
                    int sd = pool2_d_start + pd;
                    int sh = pool2_h_start + ph;
                    int sw = pool2_w_start + pw;

                    if (sd >= post_pool1_depth || sh >= post_pool1_height || sw >= post_pool1_width) continue;

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
                                                int in_d = dd - padding + kd;
                                                int in_h = hh - padding + kh;
                                                int in_w = ww - padding + kw;

                                                if (in_d < 0 || in_h < 0 || in_w < 0 || 
                                                    in_d >= in_depth || in_h >= in_height || in_w >= in_width) continue;

                                                int weight_idx = get_index(kd, kh, kw, kernel_size, kernel_size, kernel_size) + 
                                                                 c * kernel_size * kernel_size * kernel_size +
                                                                 oc * in_channels * kernel_size * kernel_size * kernel_size;
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

                                if (conv_val > local_max) {
                                    local_max = conv_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        final_sum += local_max;
    }

    output[idx] = final_sum;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_transpose_maxpool_sum(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels, int kernel_size, int stride, int padding,
    int in_depth, int in_height, int in_width
);
"""

# Calculate output dimensions
def calculate_conv_transpose_output(in_size, kernel_size, stride, padding):
    return (in_size - 1) * stride - 2 * padding + kernel_size

def calculate_maxpool_output(in_size, kernel_size, stride=None):
    if stride is None:
        stride = kernel_size
    return (in_size - kernel_size) // stride + 1

# Given parameters
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size_val = 5
stride_val = 2
padding_val = 2

# Compute intermediate shapes
d1 = calculate_conv_transpose_output(depth, kernel_size_val, stride_val, padding_val)
h1 = calculate_conv_transpose_output(height, kernel_size_val, stride_val, padding_val)
w1 = calculate_conv_transpose_output(width, kernel_size_val, stride_val, padding_val)

d2 = calculate_maxpool_output(d1, 2)
h2 = calculate_maxpool_output(h1, 2)
w2 = calculate_maxpool_output(w1, 2)

d3 = calculate_maxpool_output(d2, 3)
h3 = calculate_maxpool_output(h2, 3)
w3 = calculate_maxpool_output(w2, 3)

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