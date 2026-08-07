import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + Add + HardSwish
fused_conv_transpose_add_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// HardSwish activation: x * relu6(x + 3) / 6
__device__ float hardswish(float x) {
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6 / 6.0f;
}

// Fused kernel: performs transposed convolution (with weight, bias), adds add_input, then applies HardSwish
extern "C" __global__ void fused_conv_transpose3d_add_hardswish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* add_input,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions
    int out_channels, int out_depth, int out_height, int out_width,
    // Conv params
    int kernel_size, int stride, int padding, int output_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % out_channels;
    int n = idx / (out_width * out_height * out_depth * out_channels);

    float sum = 0.0f;

    // Loop over input channels and kernel positions
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Compute input feature map position
                    int i_d = d - stride * (kd - padding);
                    int i_h = h - stride * (kh - padding);
                    int i_w = w - stride * (kw - padding);

                    // Adjust for output padding (distributed to both sides as much as possible)
                    int offset_d = (d >= (out_depth - output_padding)) ? 1 : 0;
                    int offset_h = (h >= (out_height - output_padding)) ? 1 : 0;
                    int offset_w = (w >= (out_width - output_padding)) ? 1 : 0;

                    i_d -= offset_d;
                    i_h -= offset_h;
                    i_w -= offset_w;

                    // Check bounds
                    if (i_d >= 0 && i_d < in_depth &&
                        i_h >= 0 && i_h < in_height &&
                        i_w >= 0 && i_w < in_width) {
                        int input_idx = ((n * in_channels + ic) * in_depth + i_d) * in_height * in_width + i_h * in_width + i_w;
                        int weight_idx = ((c * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias and add_input
    int add_idx = ((n * out_channels + c) * out_depth + d) * out_height * out_width + h * out_width + w;
    sum += bias[c] + add_input[add_idx];

    // Apply HardSwish
    output[idx] = hardswish(sum);
}
"""

fused_conv_transpose_add_hardswish_cpp = """
torch::Tensor fused_conv_transpose3d_add_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int output_padding,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_transpose_add_hardswish",
    cpp_sources=[fused_conv_transpose_add_hardswish_cpp],
    cuda_sources=[fused_conv_transpose_add_hardswish_source],
    functions=["fused_conv_transpose3d_add_hardswish_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused 3D transposed convolution, addition, and HardSwish activation in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x, add_input):
        batch_size, _, in_depth, in_height, in_width = x.shape
        out_depth = (in_depth - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        output = fused_op.fused_conv_transpose3d_add_hardswish_cuda(
            x, self.weight, self.bias.squeeze(), add_input,
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.output_padding,
            in_depth, in_height, in_width, out_depth, out_height, out_width
        )
        return output