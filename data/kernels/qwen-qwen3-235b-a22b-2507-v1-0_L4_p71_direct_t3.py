import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + Divide + LeakyReLU
fused_conv_div_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_div_leakyrelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int out_channels,
    int in_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    float divisor,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % out_channels;
    int b = idx / (output_width * output_height * out_channels);

    float val = 0.0f;
    int k_half = kernel_size / 2;
    
    for (int ci = 0; ci < in_channels; ci++) {
        for (int ki = 0; ki < kernel_size; ki++) {
            for (int kj = 0; kj < kernel_size; kj++) {
                int ii = h + ki - k_half;
                int jj = w + kj - k_half;
                
                if (ii >= 0 && ii < input_height && jj >= 0 && jj < input_width) {
                    int input_idx = b * in_channels * input_height * input_width +
                                    ci * input_height * input_width +
                                    ii * input_width + jj;
                    int weight_idx = c * in_channels * kernel_size * kernel_size +
                                     ci * kernel_size * kernel_size +
                                     ki * kernel_size + kj;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    val += bias[c];
    val = val / divisor;
    val = (val > 0) ? val : negative_slope * val;
    
    output[idx] = val;
}

torch::Tensor fused_conv_div_leakyrelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    float negative_slope,
    int output_height,
    int output_width
) {
    auto batch_size = input.size(0);
    auto out_channels = bias.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto kernel_size = weight.size(3);

    auto output = torch::empty(
        {batch_size, out_channels, output_height, output_width},
        input.options()
    );

    int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_div_leakyrelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        in_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        divisor,
        negative_slope
    );

    return output;
}
"""

fused_conv_div_leakyrelu_cpp_source = """
torch::Tensor fused_conv_div_leakyrelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    float negative_slope,
    int output_height,
    int output_width
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_div_leakyrelu",
    cpp_sources=fused_conv_div_leakyrelu_cpp_source,
    cuda_sources=fused_conv_div_leakyrelu_source,
    functions=["fused_conv_div_leakyrelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Conv2d + Divide + LeakyReLU in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.divisor = divisor
        self.negative_slope = 0.01
        self.fused_kernel = fused_op.fused_conv_div_leakyrelu_cuda
        self.kernel_size = kernel_size

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        kernel_size_val = self.kernel_size
        padding = kernel_size_val // 2
        output_height = x.size(2) + 2 * padding - kernel_size_val + 1
        output_width = x.size(3) + 2 * padding - kernel_size_val + 1
        return self.fused_kernel(x, weight, bias, self.divisor, self.negative_slope, output_height, output_width)