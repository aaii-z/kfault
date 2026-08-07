import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused transposed conv + global avg pool + bias add + logsumexp + sum + scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

extern "C" __global__ void fused_transpose_conv_gap_lse_sum_scale_kernel(
    const float* input,
    const float* weight,
    const float* bias_vec,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int pad,
    float scale_factor
) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;

    int shared_mem_size = out_channels * sizeof(float);
    extern __shared__ float shared_mem[];
    float* channel_buf = shared_mem;

    // Initialize channel buffer
    for (int oc = 0; oc < out_channels; oc++) {
        float conv_val = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    float w = weight[oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + ky * kernel_size + kx];
                    float acc = 0.0f;
                    for (int h = 0; h < input_height; h++) {
                        for (int w_idx = 0; w_idx < input_width; w_idx++) {
                            int ih = h;
                            int iw = w_idx;
                            int oh = ih + ky - pad;
                            int ow = iw + kx - pad;
                            if (oh >= 0 && oh < output_height && ow >= 0 && ow < output_width) {
                                acc += input[batch * in_channels * input_height * input_width + ic * input_height * input_width + ih * input_width + iw];
                            }
                        }
                    }
                    conv_val += w * acc;
                }
            }
        }
        // Add bias
        conv_val += bias_vec[oc];
        channel_buf[oc] = conv_val;
    }

    // Synchronize threads
    __syncthreads();

    // Perform logsumexp across channels
    float max_val = -INFINITY;
    for (int oc = 0; oc < out_channels; oc++) {
        max_val = MAX(max_val, channel_buf[oc]);
    }

    float sum_exp = 0.0f;
    for (int oc = 0; oc < out_channels; oc++) {
        sum_exp += expf(channel_buf[oc] - max_val);
    }
    float logsumexp = max_val + logf(sum_exp);

    // Final scaling (GAP is implicit, sum over spatial is 1x1, so just scale)
    float result = logsumexp * scale_factor;

    if (threadIdx.x == 0) {
        output[batch] = result;
    }
}

extern "C" void fused_transpose_conv_gap_lse_sum_scale_launcher(
    const float* input,
    const float* weight,
    const float* bias_vec,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int pad,
    float scale_factor
) {
    int block_size = 256;
    int grid_size = batch_size;
    int shared_mem_size = out_channels * sizeof(float);

    fused_transpose_conv_gap_lse_sum_scale_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input, weight, bias_vec, output,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size, pad, scale_factor
    );
}
"""

fused_cpp_source = """
#include <torch/extension.h>

void fused_transpose_conv_gap_lse_sum_scale_launcher(
    const float* input,
    const float* weight,
    const float* bias_vec,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int pad,
    float scale_factor
);

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int output_height,
    int output_width,
    int kernel_size,
    int pad,
    float scale_factor
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    auto out_channels = weight.size(0);

    auto output = torch::zeros({batch_size}, dtype=torch::kFloat32).cuda();

    fused_transpose_conv_gap_lse_sum_scale_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        pad,
        scale_factor
    );

    return output;
}
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_transpose_conv_gap_lse_sum_scale",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_op_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel replacing:
    - Transposed Convolution
    - Global Average Pooling
    - Bias Addition
    - LogSumExp
    - Sum over spatial
    - Scalar multiplication
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias_shape = bias_shape
        self.pad = kernel_size // 2
        self.scale_factor = 10.0

        # Initialize transposed convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Precompute output dimensions
        self.output_height = 512 + kernel_size - 1 - 2 * self.pad
        self.output_width = 512 + kernel_size - 1 - 2 * self.pad

        # Register fused CUDA op
        self.fused_op = fused_op

    def forward(self, x):
        batch_size = x.size(0)
        output = self.fused_op.fused_op_cuda(
            x,
            self.weight,
            self.bias.squeeze(),
            self.output_height,
            self.output_width,
            self.kernel_size,
            self.pad,
            self.scale_factor
        )
        return output