import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused transposed convolution, scaling, and double global average pooling
fused_conv_transpose_gap_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Fused kernel: ConvTranspose2d -> Scale -> GAP -> GAP
// We compute the transposed convolution output on-the-fly, apply scaling, then directly reduce spatially.

__global__ void fused_conv_transpose_gap_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float multiplier
) {
    int batch_idx = blockIdx.x;
    int out_c = blockIdx.y;

    if (batch_idx >= batch_size || out_c >= out_channels) return;

    float sum = 0.0f;

    for (int c = 0; c < in_channels; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                float w = weight[out_c * in_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + ky * kernel_size + kx];
                for (int ih = 0; ih < in_height; ++ih) {
                    for (int iw = 0; iw < in_width; ++iw) {
                        int oh = ih * stride - padding + ky;
                        int ow = iw * stride - padding + kx;

                        // Apply output padding only at the last position
                        if (ky == kernel_size - 1 && kx == kernel_size - 1) {
                            if (ih == in_height - 1 && iw == in_width - 1) {
                                oh += output_padding;
                                ow += output_padding;
                            }
                        }

                        if (oh < 0 || oh >= out_height || ow < 0 || ow >= out_width) continue;

                        float val = input[batch_idx * in_channels * in_height * in_width + c * in_height * in_width + ih * in_width + iw];
                        sum += val * w;
                    }
                }
            }
        }
    }

    sum += bias[out_c];
    sum *= multiplier;

    // Perform global average pooling over spatial dimensions (already reduced to scalar per channel)
    float spatial_area = out_height * out_width;
    output[batch_idx * out_channels + out_c] = sum / spatial_area;
    // Second GAP over (1,1) is a no-op
}
"""

fused_conv_transpose_gap_cpp_source = """
torch::Tensor fused_conv_transpose_gap_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float multiplier,
    int stride,
    int padding,
    int output_padding,
    int out_height,
    int out_width);
"""

fused_conv_transpose_gap_source += """
torch::Tensor fused_conv_transpose_gap_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float multiplier,
    int stride,
    int padding,
    int output_padding,
    int out_height,
    int out_width) {

    auto options = input.options();
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels}, options);

    dim3 grid(batch_size, out_channels);
    fused_conv_transpose_gap_kernel<<<grid, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        multiplier
    );

    // Reshape to (B, C, 1, 1)
    return output.view({batch_size, out_channels, 1, 1});
}
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_transpose_gap",
    cpp_sources=[fused_conv_transpose_gap_cpp_source],
    cuda_sources=[fused_conv_transpose_gap_source],
    functions=["fused_conv_transpose_gap_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.multiplier = multiplier

        # Initialize transposed convolution parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

        # Bind the fused CUDA function
        self.fused_conv_transpose_gap = fused_op.fused_conv_transpose_gap_cuda

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        return self.fused_conv_transpose_gap(
            x,
            self.weight,
            self.bias,
            self.multiplier,
            self.stride,
            self.padding,
            self.output_padding,
            out_height,
            out_width
        )