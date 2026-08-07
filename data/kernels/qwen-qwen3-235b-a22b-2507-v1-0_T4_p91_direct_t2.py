import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose2d + Softmax + Bias + Scaling + Sigmoid
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Efficient fused kernel for transposed convolution + softmax + bias + scale + sigmoid
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

    int spatial_idx = idx % (out_height * out_width);
    int w = spatial_idx % out_width;
    int h = spatial_idx / out_width;
    int c = (idx / (out_height * out_width)) % out_channels;
    int b = idx / (out_channels * out_height * out_width);

    // Transposed convolution computation
    float conv_val = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ki = 0; ki < kernel_size; ++ki) {
            for (int kj = 0; kj < kernel_size; ++kj) {
                int i_h = (h + padding - ki) / stride;
                int i_w = (w + padding - kj) / stride;
                int remainder_h = (h + padding - ki) % stride;
                int remainder_w = (w + padding - kj) % stride;

                if (remainder_h == 0 && remainder_w == 0 &&
                    i_h >= 0 && i_h < in_height &&
                    i_w >= 0 && i_w < in_width) {
                    int input_idx = b * in_channels * in_height * in_width +
                                    ic * in_height * in_width +
                                    i_h * in_width + i_w;
                    int weight_idx = c * in_channels * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size +
                                     ki * kernel_size + kj;
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Store conv result in output temporarily for softmax reduction
    int output_idx = b * out_channels * out_height * out_width + c * out_height * out_width + spatial_idx;
    output[output_idx] = conv_val;

    __syncthreads();

    // Perform softmax across channels for this spatial location
    if (threadIdx.x == 0) {
        float max_val = -FLT_MAX;
        for (int cc = 0; cc < out_channels; ++cc) {
            int temp_idx = b * out_channels * out_height * out_width + cc * out_height * out_width + spatial_idx;
            max_val = fmaxf(max_val, output[temp_idx]);
        }

        float sum = 0.0f;
        for (int cc = 0; cc < out_channels; ++cc) {
            int temp_idx = b * out_channels * out_height * out_width + cc * out_height * out_width + spatial_idx;
            float exp_val = expf(output[temp_idx] - max_val);
            sum += exp_val;
            output[temp_idx] = exp_val;
        }

        for (int cc = 0; cc < out_channels; ++cc) {
            int temp_idx = b * out_channels * out_height * out_width + cc * out_height * out_width + spatial_idx;
            float softmax_val = output[temp_idx] / (sum + 1e-10f);
            float biased = softmax_val + bias_vec[cc];
            float scaled = biased * scaling_factor;
            float sigmoid_out = 1.0f / (1.0f + expf(-scaled));
            output[temp_idx] = sigmoid_out;
        }
    }

    __syncthreads();
}

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
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias_vec);

    int batch_size = input.size(0);

    auto output = torch::empty(
        {batch_size, out_channels, out_height, out_width},
        input.options()
    );

    int total_elements = batch_size * out_channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    dim3 grid(grid_size);
    dim3 block(block_size);

    fused_conv_softmax_bias_sigmoid_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_vec.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_height,
        out_width,
        in_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        scaling_factor
    );

    return output;
}
"""

fused_cpp_source = """
#include <torch/extension.h>
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
    extra_cflags=["-O3"]
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