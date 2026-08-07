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

    // Transposed convolution: reverse mapping from output to input
    float sum = 0.0f;

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int i_d = (d + stride * (kd - padding) - output_padding / 2) / stride;
                int i_h = (h + stride * (kh - padding) - output_padding / 2) / stride;
                int i_w = (w + stride * (kw - padding) - output_padding / 2) / stride;

                // Check if the computed input index is valid and corresponds to the correct output position
                if (i_d >= 0 && i_d < in_depth && i_h >= 0 && i_h < in_height && i_w >= 0 && i_w < in_width) {
                    // Check if this input location contributes to the current output location
                    int od_start = stride * i_d - padding;
                    int oh_start = stride * i_h - padding;
                    int ow_start = stride * i_w - padding;

                    int od_end = od_start + kernel_size;
                    int oh_end = oh_start + kernel_size;
                    int ow_end = ow_start + kernel_size;

                    if (d >= od_start && d < od_end && h >= oh_start && h < oh_end && w >= ow_start && w < ow_end) {
                        int kd_check = d - od_start;
                        int kh_check = h - oh_start;
                        int kw_check = w - ow_start;

                        if (kd_check >= 0 && kd_check < kernel_size && kh_check >= 0 && kh_check < kernel_size && kw_check >= 0 && kw_check < kernel_size) {
                            for (int ic = 0; ic < in_channels; ++ic) {
                                int input_idx = ((n * in_channels + ic) * in_depth + i_d) * in_height * in_width + i_h * in_width + i_w;
                                int weight_idx = ((c * in_channels + ic) * kernel_size + kd_check) * kernel_size * kernel_size + kh_check * kernel_size + kw_check;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
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

// CUDA C++ wrapper function
torch::Tensor fused_conv_transpose3d_add_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int output_padding,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width
) {
    auto output = torch::empty({input.size(0), out_channels, out_depth, out_height, out_width}, input.options());

    const int total_elements = output.numel();
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_conv_transpose3d_add_hardswish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0), in_channels, in_depth, in_height, in_width,
        out_channels, out_depth, out_height, out_width,
        kernel_size, stride, padding, output_padding
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return output;
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