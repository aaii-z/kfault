import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + Scale + Min along channel dimension
fused_conv_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// CUDA kernel to perform fused conv2d, scaling, and channel-wise min reduction
__global__ void fused_conv_scale_min_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    int oy = blockIdx.y;
    int ox = blockIdx.z;

    if (batch_idx >= batch_size || oy >= output_height || ox >= output_width) return;

    int kernel_radius = kernel_size / 2;
    float min_val = FLT_MAX;

    for (int oc = 0; oc < out_channels; ++oc) {
        float conv_val = bias[oc];  // Start with bias

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int iy = oy + ky - kernel_radius;
                    int ix = ox + kx - kernel_radius;

                    if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
                        int input_idx = batch_idx * in_channels * input_height * input_width +
                                        ic * input_height * input_width +
                                        iy * input_width + ix;
                        int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                         ic * kernel_size * kernel_size +
                                         ky * kernel_size + kx;
                        conv_val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Scale after convolution
        conv_val *= scale_factor;

        // Update min across channels
        min_val = fminf(min_val, conv_val);
    }

    // Output is (batch, 1, H, W) - keepdim=True equivalent
    int output_idx = batch_idx * 1 * output_height * output_width +
                     0 * output_height * output_width +
                     oy * output_width + ox;
    output[output_idx] = min_val;
}
"""

fused_conv_scale_min_cpp_source = """
torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_scale_min",
    cpp_sources=fused_conv_scale_min_cpp_source,
    cuda_sources=fused_conv_scale_min_source,
    functions=["fused_conv_scale_min_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + Scale + Min along channel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Initialize weights and bias as in Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

        # Fused CUDA function
        self.fused_op = fused_op.fused_conv_scale_min_cuda

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias, self.scale_factor)