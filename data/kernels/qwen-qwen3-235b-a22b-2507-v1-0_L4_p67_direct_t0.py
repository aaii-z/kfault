import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + GELU + Adaptive Average Pooling to 1x1
fused_conv_gelu_gap_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation approximation: x * 0.5 * (1 + erf(x / sqrt(2)))
__device__ float gelu(float x) {
    return x * 0.5f * (1.0f + ::erff(x * 0.707106781f));
}

__global__ void fused_conv_gelu_gap_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int output_height,
    int output_width
) {
    int batch = blockIdx.x;
    int oc = blockIdx.y;

    if (batch >= batch_size || oc >= out_channels) return;

    float sum = 0.0f;
    int kh = kernel_size;
    int kw = kernel_size;
    int ih = input_height;
    int iw = input_width;
    int ic, ky, kx, iy, ix;

    // Convolution + GELU in one loop, accumulate spatial dims directly for global average pool
    for (ic = 0; ic < in_channels; ic++) {
        for (ky = 0; ky < kh; ky++) {
            for (kx = 0; kx < kw; kx++) {
                float w = weight[oc * in_channels * kh * kw + ic * kh * kw + ky * kw + kx];
                float b_val = (kx == 0 && ky == 0) ? bias[oc] : 0.0f; // Add bias once per output channel
                float conv_val = 0.0f;
                for (iy = 0; iy < output_height; iy++) {
                    for (ix = 0; ix < output_width; ix++) {
                        int i_offset = batch * in_channels * ih * iw + ic * ih * iw + (iy + ky) * iw + (ix + kx);
                        conv_val += input[i_offset] * w;
                    }
                }
                sum += conv_val + b_val;
            }
        }
    }

    // Apply GELU
    sum = gelu(sum);

    // Since we are doing adaptive_avg_pool2d(x, 1), we just need one output value per channel
    // But note: in standard conv -> gelu -> gap, we apply GELU *before* pooling.
    // So correct order: conv -> gelu -> avg pool.
    // We need to apply GELU element-wise on the conv output feature map, then average.

    // Let's recompute properly: first compute full conv output, apply GELU per element, then average.
    // We'll do it in two passes implicitly via shared memory or just restructure.

    // Due to complexity of fusing all three with proper semantics, we instead:
    // 1. Compute conv output with bias (H x W)
    // 2. Apply GELU on each element
    // 3. Sum all elements and divide by H*W

    // We'll recompute with correct order
    float acc = 0.0f;
    for (iy = 0; iy < output_height; iy++) {
        for (ix = 0; ix < output_width; ix++) {
            float conv_val = bias[oc];
            for (ic = 0; ic < in_channels; ic++) {
                for (ky = 0; ky < kh; ky++) {
                    for (kx = 0; kx < kw; kx++) {
                        int iy_pad = iy + ky;
                        int ix_pad = ix + kx;
                        if (iy_pad < ih && ix_pad < iw) {
                            int i_offset = batch * in_channels * ih * iw + ic * ih * iw + iy_pad * iw + ix_pad;
                            int w_offset = oc * in_channels * kh * kw + ic * kh * kw + ky * kw + kx;
                            conv_val += input[i_offset] * weight[w_offset];
                        }
                    }
                }
            }
            // Apply GELU
            float gelu_val = gelu(conv_val);
            acc += gelu_val;
        }
    }

    // Global average pool: divide by spatial size
    acc /= (output_height * output_width);

    // Write output
    output[batch * out_channels + oc] = acc;
}
"""

fused_conv_gelu_gap_cpp = """
torch::Tensor fused_conv_gelu_gap_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t input_height,
    int64_t input_width
);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_conv_gelu_gap",
    cpp_sources=[fused_conv_gelu_gap_cpp],
    cuda_sources=[fused_conv_gelu_gap_source],
    functions=["fused_conv_gelu_gap_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model using fused CUDA kernel for Conv2d + GELU + AdaptiveAvgPool2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

        # Bind the fused CUDA function
        self.fused_op = fused_ops.fused_conv_gelu_gap_cuda

    def forward(self, x):
        batch_size = x.size(0)
        input_height = x.size(2)
        input_width = x.size(3)
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1

        return self.fused_op(
            x,
            self.weight,
            self.bias,
            self.out_channels,
            self.kernel_size,
            input_height,
            input_width
        )