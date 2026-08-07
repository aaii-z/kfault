import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

fused_conv_transpose_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_gelu_kernel(
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
    int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    if (idx >= total_elements) return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c_out = (idx / (output_width * output_height)) % out_channels;
    int b = idx / (output_width * output_height * out_channels);

    float sum_val = 0.0f;
    const int kernel_size = 3;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                int h_in = h_out - k_h;
                int w_in = w_out - k_w;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int input_idx = ((b * in_channels + c_in) * input_height + h_in) * input_width + w_in;
                    int weight_idx = ((c_in * out_channels + c_out) * kernel_size + k_h) * kernel_size + k_w;
                    sum_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    sum_val += bias[c_out];
    
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (sum_val + coeff * sum_val * sum_val * sum_val);
    output[idx] = 0.5f * sum_val * (1.0f + tanhf(inner));
}

torch::Tensor fused_conv_transpose_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    auto batch_size = input.size(0);
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_transpose_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width
    );

    return output;
}
"""

fused_conv_transpose_gelu_cpp = "torch::Tensor fused_conv_transpose_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int in_channels, int out_channels, int input_height, int input_width, int output_height, int output_width);"

fused_conv_transpose_gelu = load_inline(
    name="fused_conv_transpose_gelu",
    cpp_sources=fused_conv_transpose_gelu_cpp,
    cuda_sources=fused_conv_transpose_gelu_source,
    functions=["fused_conv_transpose_gelu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses transposed convolution and GELU into a single CUDA kernel,
    followed by GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.fused_conv_transpose_gelu = fused_conv_transpose_gelu

    def forward(self, x):
        stride = self.conv_transpose.stride[0]
        padding = self.conv_transpose.padding[0]
        output_padding = self.conv_transpose.output_padding[0]
        kernel_size = self.conv_transpose.kernel_size[0]
        
        input_height = x.size(2)
        input_width = x.size(3)
        output_height = (input_height - 1) * stride + kernel_size - 2 * padding + output_padding
        output_width = (input_width - 1) * stride + kernel_size - 2 * padding + output_padding

        x = self.fused_conv_transpose_gelu.fused_conv_transpose_gelu_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.in_channels,
            self.conv_transpose.out_channels,
            input_height,
            input_width,
            output_height,
            output_width
        )
        x = self.group_norm(x)
        return x