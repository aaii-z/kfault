import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + Mish activation (applied twice)
conv2d_mish_mish_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Mish activation function: f(x) = x * tanh(softplus(x))
__device__ float mish(float x) {
    return x * tanh(log1p(exp(x)));
}

__global__ void conv2d_mish_mish_kernel(
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
    int pad,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    int input_h_start = h * stride - pad;
    int input_w_start = w * stride - pad;

    float conv_val = 0.0f;

    for (int ci = 0; ci < in_channels; ++ci) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = input_h_start + kh;
                int iw = input_w_start + kw;

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

    conv_val += bias[c];
    float mish1 = mish(conv_val);
    float mish2 = mish(mish1);
    output[idx] = mish2;
}

torch::Tensor conv2d_mish_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int out_channels,
    int in_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int pad,
    int stride) {
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    conv2d_mish_mish_kernel<<<grid_size, block_size>>>(
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
        pad,
        stride
    );

    return output;
}
"""

conv2d_mish_mish_cpp_source = """
torch::Tensor conv2d_mish_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int out_channels,
    int in_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int pad,
    int stride);
"""

# Compile the inline CUDA kernel
conv2d_mish_mish_module = load_inline(
    name="conv2d_mish_mish",
    cpp_sources=conv2d_mish_mish_cpp_source,
    cuda_sources=conv2d_mish_mish_cuda_source,
    functions=["conv2d_mish_mish_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused Conv2d + Mish + Mish in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.stride = 1

        # Create the convolution layer to keep weight/bias parameters
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.pad, stride=self.stride)

        # Register the fused CUDA kernel
        self.fused_op = conv2d_mish_mish_module.conv2d_mish_mish_cuda

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        batch_size, _, input_height, input_width = x.size()
        output_height = (input_height + 2 * self.pad - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.pad - self.kernel_size) // self.stride + 1

        return self.fused_op(
            x, weight, bias,
            batch_size, self.out_channels, self.in_channels,
            input_height, input_width,
            output_height, output_width,
            self.kernel_size, self.pad, self.stride
        )