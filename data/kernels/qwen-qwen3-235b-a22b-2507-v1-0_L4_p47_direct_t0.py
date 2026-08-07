import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Conv3D + Mish + Tanh
fused_conv_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_conv3d_mish_tanh_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size_d,
    int kernel_size_h,
    int kernel_size_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w
) {
    int output_size = output_depth * output_height * output_width;
    int output_elem_count = batch_size * out_channels * output_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= output_elem_count) return;

    int temp = idx;
    int w = temp % output_width; temp /= output_width;
    int h = temp % output_height; temp /= output_height;
    int d = temp % output_depth; temp /= output_depth;
    int c_out = temp % out_channels; temp /= out_channels;
    int b = temp;

    int input_idx_base = b * in_channels * input_depth * input_height * input_width;
    int weight_idx_base = c_out * in_channels * kernel_size_d * kernel_size_h * kernel_size_w;
    int output_idx = idx;

    float acc = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size_d; ++kd) {
            for (int kh = 0; kh < kernel_size_h; ++kh) {
                for (int kw = 0; kw < kernel_size_w; ++kw) {
                    int i_d = d * stride_d + kd - pad_d;
                    int i_h = h * stride_h + kh - pad_h;
                    int i_w = w * stride_w + kw - pad_w;

                    if (i_d >= 0 && i_d < input_depth && i_h >= 0 && i_h < input_height && i_w >= 0 && i_w < input_width) {
                        int input_idx = input_idx_base + c_in * input_depth * input_height * input_width + i_d * input_height * input_width + i_h * input_width + i_w;
                        int weight_idx = weight_idx_base + c_in * kernel_size_d * kernel_size_h * kernel_size_w + kd * kernel_size_h * kernel_size_w + kh * kernel_size_w + kw;
                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    acc += bias[c_out];
    acc = tanhf(mish_activation(acc));
    output[output_idx] = acc;
}

torch::Tensor fused_conv3d_mish_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int kernel_size_d, int kernel_size_h, int kernel_size_w
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);

    auto out_channels = weight.size(0);
    auto output_depth = (input_depth + 2 * pad_d - kernel_size_d) / stride_d + 1;
    auto output_height = (input_height + 2 * pad_h - kernel_size_h) / stride_h + 1;
    auto output_width = (input_width + 2 * pad_w - kernel_size_w) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int output_elem_count = batch_size * out_channels * output_depth * output_height * output_width;
    const int block_size = 256;
    const int grid_size = (output_elem_count + block_size - 1) / block_size;

    dim3 block(block_size);
    dim3 grid(grid_size);

    fused_conv3d_mish_tanh_forward<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size_d,
        kernel_size_h,
        kernel_size_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w
    );

    return output;
}
"""

fused_conv_mish_tanh_cpp = """
torch::Tensor fused_conv3d_mish_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int kernel_size_d, int kernel_size_h, int kernel_size_w
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_mish_tanh",
    cpp_sources=fused_conv_mish_tanh_cpp,
    cuda_sources=fused_conv_mish_tanh_source,
    functions=["fused_conv3d_mish_tanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses 3D convolution, Mish activation, and Tanh activation into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        weight = self.conv.weight
        bias = self.conv.bias
        kernel_size_d, kernel_size_h, kernel_size_w = self.kernel_size, self.kernel_size, self.kernel_size
        stride_d, stride_h, stride_w = self.stride, self.stride, self.stride
        pad_d, pad_h, pad_w = self.padding, self.padding, self.padding

        return self.fused_op.fused_conv3d_mish_tanh_cuda(
            x, weight, bias,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            kernel_size_d, kernel_size_h, kernel_size_w
        )