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

// Shared memory for softmax reduction
extern __shared__ float sdata[];

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
    int spatial_idx = blockIdx.x;
    int total_spatial = out_height * out_width;
    if (spatial_idx >= total_spatial) return;

    int w = spatial_idx % out_width;
    int h = spatial_idx / out_width;
    int tid = threadIdx.x;
    int b = blockIdx.y;
    int n_channels = out_channels;

    // Each block handles one spatial location (h, w) for all channels and one batch
    float conv_val = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ki = 0; ki < kernel_size; ++ki) {
            for (int kj = 0; kj < kernel_size; ++kj) {
                int in_h = h * stride - padding + ki;
                int in_w = w * stride - padding + kj;

                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    int input_idx = b * in_channels * in_height * in_width +
                                    ic * in_height * in_width +
                                    in_h * in_width + in_w;
                    int weight_idx = ic * n_channels * kernel_size * kernel_size +  // (ic, c, ki, kj)
                                     tid * kernel_size * kernel_size +
                                     ki * kernel_size + kj;
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Store conv result in shared memory for softmax
    sdata[tid] = conv_val;
    __syncthreads();

    // Compute max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = tid; i < n_channels; i += blockDim.x) {
        max_val = fmaxf(max_val, sdata[i]);
    }
    max_val = max_val;
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    __syncthreads();

    // Compute exp(x - max) and sum
    float exp_val = expf(sdata[tid] - max_val);
    sdata[tid] = exp_val;
    __syncthreads();

    // Block-wide sum reduction
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, exp_val, offset);
        exp_val += other;
    }
    float sum = exp_val;
    if (threadIdx.x == 0) {
        sdata[0] = sum;
    }
    __syncthreads();
    sum = sdata[0];

    // Final softmax, bias, scale, sigmoid
    float softmax_val = exp_val / (sum + 1e-10f);
    float biased = softmax_val + bias_vec[tid];
    float scaled = biased * scaling_factor;
    float sigmoid_out = 1.0f / (1.0f + expf(-scaled));
    int output_idx = b * n_channels * total_spatial + tid * total_spatial + spatial_idx;
    output[output_idx] = sigmoid_out;
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
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
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