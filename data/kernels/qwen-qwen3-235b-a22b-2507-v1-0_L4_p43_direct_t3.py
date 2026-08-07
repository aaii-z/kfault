import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv3d + MaxPool3d + LogSumExp + ReLU with high parallelism
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

// Shared memory for logsumexp reduction
extern __shared__ float sdata[];

__global__ void fused_conv_maxpool_logsumexp_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions after conv and maxpool
    int out_channels, int out_depth, int out_height, int out_width,
    // Conv parameters
    int kernel_size, int conv_stride, int padding,
    // Precomputed sizes
    int conv_out_depth, int conv_out_height, int conv_out_width
) {
    int batch = blockIdx.x;
    int od = blockIdx.y;
    int oh = blockIdx.z / out_width;
    int ow = blockIdx.z % out_width;

    if (batch >= batch_size || od >= out_depth || oh >= out_height || ow >= out_width) return;

    // Each thread block handles one output spatial location (od, oh, ow)
    // We use thread 0 to compute the 8 max values from 2x2x2 maxpool region
    float max_vals[8];
    for (int i = 0; i < 8; ++i) max_vals[i] = -FLT_MAX;

    // Compute starting spatial location in the conv output (before maxpool)
    int conv_d_start = od * 2;
    int conv_h_start = oh * 2;
    int conv_w_start = ow * 2;

    // Use all threads in block to compute conv for different output channels
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Iterate over output channels in a strided manner
    float local_max[8];
    for (int i = 0; i < 8; ++i) local_max[i] = -FLT_MAX;

    for (int oc = tid; oc < out_channels; oc += num_threads) {
        float conv_val = 0.0f;

        // Convolution over input channels and kernel
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                int id = conv_d_start + kd - padding;
                if (id < 0 || id >= in_depth) continue;
                for (int kh = 0; kh < kernel_size; ++kh) {
                    int ih = conv_h_start + kh - padding;
                    if (ih < 0 || ih >= in_height) continue;
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int iw = conv_w_start + kw - padding;
                        if (iw < 0 || iw >= in_width) continue;

                        int in_idx = get_index(batch, ic, id, ih, iw,
                                               batch_size, in_channels, in_depth, in_height, in_width);
                        int w_idx = ((oc * in_channels + ic) * kernel_size + kd) * (kernel_size * kernel_size) +
                                    kh * kernel_size + kw;
                        conv_val += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        conv_val += bias[oc];

        // Determine local position within 2x2x2 maxpool grid
        int local_d = (conv_d_start) & 1;
        int local_h = (conv_h_start) & 1;
        int local_w = (conv_w_start) & 1;
        int local_idx = (local_d << 2) | (local_h << 1) | local_w;

        // Update max value for this bin
        if (conv_val > local_max[local_idx]) {
            local_max[local_idx] = conv_val;
        }
    }

    // Reduce across threads: each thread writes its 8 max values
    __shared__ float shared_max[256 * 8]; // assume max 256 threads
    int base = tid * 8;
    for (int i = 0; i < 8; ++i) {
        shared_max[base + i] = local_max[i];
    }
    __syncthreads();

    // Final reduction by thread 0
    if (tid == 0) {
        for (int i = 0; i < 8; ++i) max_vals[i] = -FLT_MAX;
        for (int t = 0; t < num_threads; ++t) {
            for (int i = 0; i < 8; ++i) {
                float val = shared_max[t * 8 + i];
                if (val > max_vals[i]) max_vals[i] = val;
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
}
"""

cpp_source = """
torch::Tensor fused_conv_mp_lse_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int conv_out_depth, int conv_out_height, int conv_out_width
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
    extra_cuda_cflags=["-O3", "-use_fast_math", "--use_fast_math", "-lineinfo"]
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
            self.kernel_size, self.stride, self.padding,
            conv_out_depth, conv_out_height, conv_out_width
        )

        return output