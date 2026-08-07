import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose2d + Mish + Add + Hardtanh + Scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
__device__ float mish(float x) {
    const float sp = log1pf(expf(x)); // softplus
    return x * tanhf(sp);
}

__global__ void fused_conv_transpose_mish_add_hardtanh_scale_kernel(
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
    int stride,
    int padding,
    int output_padding,
    float add_value,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    // ConvTranspose2d: sum over input channels, kernel space
    float val = 0.0f;
    int k = kernel_size;
    int pad = padding;
    int stride_val = stride;
    int out_pad = output_padding;

    for (int ci = 0; ci < in_channels; ci++) {
        for (int y = 0; y < k; y++) {
            for (int x = 0; x < k; x++) {
                int ih = h * stride_val - pad + y;
                int iw = w * stride_val - pad + x;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = n * in_channels * input_height * input_width +
                                    ci * input_height * input_width +
                                    ih * input_width + iw;

                    int weight_idx = c * in_channels * k * k +
                                     ci * k * k + y * k + x;

                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    val += bias[c];

    // Apply Mish activation
    val = mish(val);

    // Add constant value
    val += add_value;

    // Hardtanh: clamp between -1 and 1
    val = fmaxf(-1.0f, fminf(1.0f, val));

    // Scale
    val *= scale;

    output[idx] = val;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_transpose_mish_add_hardtanh_scale_cuda(
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
    int stride,
    int padding,
    int output_padding,
    float add_value,
    float scale);
"""

fused_kernel = load_inline(
    name="fused_conv_transpose_mish_add_hardtanh_scale",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_transpose_mish_add_hardtanh_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for transposed convolution, Mish activation, 
    addition, Hardtanh activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale

        # Initialize transposed convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

        # Register the fused CUDA kernel
        self.fused_op = fused_kernel.fused_conv_transpose_mish_add_hardtanh_scale_cuda

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape

        # Compute output dimensions for transposed convolution
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        return self.fused_op(
            x, self.weight, self.bias,
            batch_size, self.out_channels, self.in_channels,
            input_height, input_width,
            output_height, output_width,
            self.kernel_size, self.stride, self.padding, self.output_padding,
            self.add_value, self.scale
        )