import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose2d + Softmax + Bias + Scaling + Sigmoid
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Efficient fused kernel for transposed conv, softmax, bias, scaling, sigmoid
__global__ void fused_conv_softmax_bias_sigmoid_kernel(
    const float* input,
    const float* weight,
    const float* bias_vec,
    float* output,
    int batch_size,
    int out_channels,
    int out_height,
    int out_width,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;

    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % out_channels;
    int b = idx / (out_width * out_height * out_channels);

    // Transposed convolution (also called deconvolution or fractionally strided conv)
    float conv_val = 0.0f;
    int kernel_radius = kernel_size / 2;
    
    for (int ki = 0; ki < kernel_size; ++ki) {
        for (int kj = 0; kj < kernel_size; ++kj) {
            for (int ic = 0; ic < in_channels; ++ic) {
                int i_h = h * stride - 2 * padding + ki;
                int i_w = w * stride - 2 * padding + kj;
                
                if (i_h >= 0 && i_h < in_height && i_w >= 0 && i_w < in_width) {
                    int input_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + i_h * in_width + i_w;
                    int weight_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + ki * kernel_size + kj;
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Store per-channel conv result
    float* conv_buffer = output + b * out_channels * out_height * out_width;
    int spatial_idx = h * out_width + w;
    int conv_out_idx = c * out_height * out_width + spatial_idx;
    conv_buffer[conv_out_idx] = conv_val;

    // Ensure all channels have been computed before proceeding to softmax
    __syncthreads();

    // Softmax across channel dimension
    float max_val = -FLT_MAX;
    for (int cc = 0; cc < out_channels; ++cc) {
        float val = conv_buffer[cc * out_height * out_width + spatial_idx];
        max_val = fmaxf(max_val, val);
    }

    float sum = 0.0f;
    float softmax_val;
    for (int cc = 0; cc < out_channels; ++cc) {
        float val = conv_buffer[cc * out_height * out_width + spatial_idx];
        float exp_val = expf(val - max_val);
        sum += exp_val;
        if (cc == c) {
            softmax_val = exp_val;
        }
    }
    softmax_val /= (sum + 1e-10);

    // Add bias, scale, and apply sigmoid
    float biased = softmax_val + bias_vec[c];
    float scaled = biased * scaling_factor;
    float sigmoid_out = 1.0f / (1.0f + expf(-scaled));

    // Write final result
    output[idx] = sigmoid_out;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_vec,
    int out_channels,
    int out_height,
    int out_width,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_op_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for transposed convolution, softmax, bias, scaling, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.fused_op = fused_op

    def forward(self, x):
        batch_size, _, in_height, in_width = x.shape
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        output = self.fused_op.fused_op_cuda(
            x,
            self.weight,
            self.bias.view(-1),
            self.out_channels,
            out_height,
            out_width,
            self.in_channels,
            in_height,
            in_width,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.scaling_factor
        )
        return output