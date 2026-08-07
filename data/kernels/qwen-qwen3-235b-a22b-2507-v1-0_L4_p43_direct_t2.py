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

    // Compute starting spatial location in the conv output (before maxpool)
    int conv_d_start = od * 2;
    int conv_h_start = oh * 2;
    int conv_w_start = ow * 2;

    // Buffer to store the 2x2x2 maxpooled values (8 values total)
    float max_vals[8];
    for (int i = 0; i < 8; ++i) {
        max_vals[i] = -FLT_MAX;
    }

    // Iterate over output channels (for convolution)
    for (int oc = 0; oc < out_channels; ++oc) {
        float conv_val = 0.0f;

        // Convolution over input channels and kernel
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
                            int w_idx = ((oc * in_channels + ic) * kernel_size + kd) * (kernel_size * kernel_size) +
                                        kh * kernel_size + kw;
                            conv_val += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        conv_val += bias[oc];

        // Determine local position within 2x2x2 maxpool grid
        int local_d = conv_d_start & 1;
        int local_h = conv_h_start & 1;
        int local_w = conv_w_start & 1;
        int local_idx = (local_d << 2) | (local_h << 1) | local_w;

        // Update max value for this bin
        if (conv_val > max_vals[local_idx]) {
            max_vals[local_idx] = conv_val;
        }
    }

    // Compute logsumexp over the 8 maxpooled values
    float max_val = -FLT_MAX;
    for (int i = 0; i < 8; ++i) {
        if (max_vals[i] > max_val) {
            max_val = max_vals[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < 8; ++i) {
        if (max_vals[i] > -FLT_MAX) {
            sum_exp += expf(max_vals[i] - max_val);
        }
    }

    float lse = max_val + logf(sum_exp);
    float relu_out = fmaxf(0.0f, lse);

    // Output has 1 channel (keepdim=True)
    int out_idx = get_index(batch, 0, od, oh, ow,
                            batch_size, 1, out_depth, out_height, out_width);
    output[out_idx] = relu_out;
}

// C++ wrapper function
torch::Tensor fused_conv_mp_lse_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding
) {
    auto output = torch::empty({batch_size, 1, out_depth, out_height, out_width}, 
                               input.options());

    const int threads_per_block = 256;
    const int num_blocks_x = batch_size;
    const int num_blocks_y = out_depth;
    const int num_blocks_z = out_height * out_width;
    dim3 grid(num_blocks_x, num_blocks_y, num_blocks_z);
    dim3 block(threads_per_block);

    fused_conv_maxpool_logsumexp_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, out_depth, out_height, out_width,
        kernel_size, stride, padding
    );

    return output;
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

        # Ensure contiguous
        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        # Launch fused kernel
        output = fused_op.fused_conv_mp_lse_relu(
            x, weight, bias,
            batch_size, self.in_channels, in_depth, in_height, in_width,
            self.out_channels, out_depth, out_height, out_width,
            self.kernel_size, self.stride, self.padding
        )

        return output