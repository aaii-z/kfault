import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused transposed convolution, bias add, clamp, scale, clamp, divide
fused_conv_transpose_bias_clamp_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utility function to get 4D index in NCHW format
__device__ int get_index(int n, int c, int h, int w, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

__global__ void fused_conv_transpose_bias_clamp_scale_divide_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding, int output_padding,
    float scaling_factor
) {
    int total_elements = batch_size * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Decode index
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);

    // Compute transposed convolution
    float sum = 0.0f;
    for (int ki = 0; ki < kernel_size; ++ki) {
        for (int kj = 0; kj < kernel_size; ++kj) {
            // Compute input position from output position
            int i_h = (h - ki + 2 * padding - output_padding) / stride;
            int i_w = (w - kj + 2 * padding - output_padding) / stride;

            // Check if the current output position is aligned with the stride
            bool is_aligned = (h - ki + 2 * padding - output_padding) >= 0 &&
                              (w - kj + 2 * padding - output_padding) >= 0 &&
                              (h - ki + 2 * padding - output_padding) % stride == 0 &&
                              (w - kj + 2 * padding - output_padding) % stride == 0;

            if (is_aligned && i_h >= 0 && i_h < in_height && i_w >= 0 && i_w < in_width) {
                for (int ci = 0; ci < in_channels; ++ci) {
                    int weight_idx = c * in_channels * kernel_size * kernel_size + ci * kernel_size * kernel_size + ki * kernel_size + kj;
                    int input_idx = get_index(n, ci, i_h, i_w, in_channels, in_height, in_width);
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias (broadcasted over spatial dimensions)
    sum += bias[c];

    // First clamp [0.0, 1.0]
    sum = fmaxf(0.0f, fminf(1.0f, sum));

    // Scale
    sum *= scaling_factor;

    // Clamp again [0.0, 1.0]
    sum = fmaxf(0.0f, fminf(1.0f, sum));

    // Divide by scaling_factor
    sum /= scaling_factor;

    output[idx] = sum;
}

torch::Tensor fused_conv_transpose_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);

    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);

    // Output spatial dimensions (from ConvTranspose2d formula)
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_transpose_bias_clamp_scale_divide_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_size, stride, padding, output_padding,
        scaling_factor
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

fused_conv_transpose_cpp_source = """
torch::Tensor fused_conv_transpose_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_conv_transpose_ops",
    cpp_sources=[fused_conv_transpose_cpp_source],
    cuda_sources=[fused_conv_transpose_bias_clamp_scale_source],
    functions=["fused_conv_transpose_op"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for transposed convolution + bias + clamps + scale + divide.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        # Initialize transposed convolution weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor
        self.fused_op = fused_ops.fused_conv_transpose_op

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias.view(-1), self.stride, self.padding, self.output_padding, self.scaling_factor)