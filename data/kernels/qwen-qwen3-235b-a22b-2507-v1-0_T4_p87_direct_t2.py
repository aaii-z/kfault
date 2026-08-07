import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d, subtraction, and Mish activation
fused_conv_subtract_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Mish activation: f(x) = x * tanh(softplus(x))
__device__ float mish_activation(float x) {
    float sp = log1pf(expf(x)); // softplus: ln(1 + exp(x))
    return x * tanhf(sp);
}

__global__ void fused_conv_subtract_mish_kernel(
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
    float subtract_value_1,
    float subtract_value_2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    int pad = kernel_size / 2;
    int kernel_radius = kernel_size / 2;

    // Compute output indices (with padding support)
    int ih_start = h - pad;
    int iw_start = w - pad;

    // Perform convolution
    float conv_val = 0.0f;
    for (int ci = 0; ci < in_channels; ++ci) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = ih_start + kh;
                int iw = iw_start + kw;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = n * in_channels * input_height * input_width +
                                    ci * input_height * input_width +
                                    ih * input_width + iw;
                    int weight_idx = c * in_channels * kernel_size * kernel_size +
                                     ci * kernel_size * kernel_size +
                                     kh * kernel_size + kw;
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias and subtract values
    float result = conv_val + bias[c] - subtract_value_1 - subtract_value_2;
    output[idx] = mish_activation(result);
}

torch::Tensor fused_conv_subtract_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value_1,
    float subtract_value_2) {
    
    // Input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    // Compute output height and width (assuming padding = kernel_size // 2, stride = 1)
    int pad = kernel_size / 2;
    int output_height = input_height;
    int output_width = input_width;

    // Output tensor
    auto output = torch::zeros(
        {batch_size, out_channels, output_height, output_width},
        torch::dtype(torch::kFloat).device(torch::kCUDA)
    );

    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    dim3 block(block_size);
    dim3 grid(grid_size);

    fused_conv_subtract_mish_kernel<<<grid, block>>>(
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
        subtract_value_1,
        subtract_value_2
    );

    return output;
}
"""

fused_conv_subtract_mish_cpp_source = """
torch::Tensor fused_conv_subtract_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value_1,
    float subtract_value_2);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_subtract_mish",
    cpp_sources=fused_conv_subtract_mish_cpp_source,
    cuda_sources=fused_conv_subtract_mish_source,
    functions=["fused_conv_subtract_mish_cuda"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Conv2d, subtraction, and Mish activation in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return fused_op.fused_conv_subtract_mish_cuda(
            x,
            self.weight,
            self.bias,
            self.subtract_value_1,
            self.subtract_value_2
        )