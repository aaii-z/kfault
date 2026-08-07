import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused transposed convolution, bias subtraction, and tanh activation
fused_conv_transpose_bias_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_bias_tanh_kernel(
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
    int output_padding
) {
    int total_elements = batch_size * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Decode index
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);

    float sum = 0.0f;

    // Loop over input channels and kernel
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Compute input position (accounting for stride and padding)
                int ih = h - kh + padding;
                int iw = w - kw + padding;

                // Check if the input position is valid and lies on the correct stride grid
                if (ih >= 0 && ih < in_height * stride && iw >= 0 && iw < in_width * stride) {
                    if (ih % stride == 0 && iw % stride == 0) {
                        ih /= stride;
                        iw /= stride;
                        int input_idx = n * in_channels * in_height * in_width + ic * in_height * in_width + ih * in_width + iw;
                        int weight_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Subtract bias and apply tanh
    sum -= bias[c];
    output[idx] = tanhf(sum);
}

torch::Tensor fused_conv_transpose_bias_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int out_height,
    int out_width) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_conv_transpose_bias_tanh_kernel<<<grid_size, block_size>>>(
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
        weight.size(2),
        stride,
        padding,
        output_padding
    );

    return output;
}
"""

fused_conv_transpose_bias_tanh_cpp_source = """
torch::Tensor fused_conv_transpose_bias_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int out_height,
    int out_width);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_transpose_bias_tanh",
    cpp_sources=fused_conv_transpose_bias_tanh_cpp_source,
    cuda_sources=fused_conv_transpose_bias_tanh_source,
    functions=["fused_conv_transpose_bias_tanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused transposed convolution, bias subtraction, and tanh activation in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op

    def forward(self, x):
        batch_size, _, in_height, in_width = x.shape

        # Calculate output spatial dimensions
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Launch fused CUDA kernel
        output = self.fused_op.fused_conv_transpose_bias_tanh_cuda(
            x, self.weight, self.bias.squeeze(), self.stride, self.padding,
            self.output_padding, out_height, out_width
        )

        return output