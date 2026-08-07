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
    int kernel_size
) {
    int batch = blockIdx.x;
    int oc = blockIdx.y;

    if (batch >= batch_size || oc >= out_channels) return;

    float sum_val = 0.0f;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int output_size = output_height * output_width;

    for (int oy = 0; oy < output_height; oy++) {
        for (int ox = 0; ox < output_width; ox++) {
            float conv_val = bias[oc];
            for (int ic = 0; ic < in_channels; ic++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int iy = oy + ky - (kernel_size / 2);
                        int ix = ox + kx - (kernel_size / 2);
                        if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
                            int i_offset = batch * in_channels * input_height * input_width +
                                          ic * input_height * input_width + iy * input_width + ix;
                            int w_offset = oc * in_channels * kernel_size * kernel_size +
                                          ic * kernel_size * kernel_size + ky * kernel_size + kx;
                            conv_val += input[i_offset] * weight[w_offset];
                        }
                    }
                }
            }
            sum_val += gelu(conv_val);
        }
    }

    // Global average pool over spatial dimensions
    sum_val /= output_size;

    // Write final output
    output[batch * out_channels + oc] = sum_val;
}

torch::Tensor fused_conv_gelu_gap_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t input_height,
    int64_t input_width
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);

    auto output = torch::empty({batch_size, out_channels}, input.options());

    dim3 block_dim(16, 16);
    dim3 grid_dim(batch_size, out_channels);

    fused_conv_gelu_gap_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size
    );

    return output;
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
        input_height = x.size(2)
        input_width = x.size(3)
        return self.fused_op(
            x,
            self.weight,
            self.bias,
            self.out_channels,
            self.kernel_size,
            input_height,
            input_width
        )