import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused transposed convolution, max pooling, hardtanh, mean, and tanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Efficient fused kernel: ConvTranspose2d -> MaxPool2d -> Hardtanh -> Mean -> Tanh
__global__ void fused_conv_transpose_maxpool_hardtanh_mean_tanh(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int input_height, int input_width,
    // Output dimensions (after conv_transpose)
    int out_channels, int out_height, int out_width,
    // Conv params
    int kernel_size, int stride, int padding, int output_padding,
    // MaxPool params
    int pool_kernel_size, int pool_stride,
    // Hardtanh params
    float hardtanh_min, float hardtanh_max,
    // Overall output spatial (1x1 after mean)
    int output_spatial_height, int output_spatial_width
) {
    int batch = blockIdx.x;
    int chan = blockIdx.y;

    if (batch >= batch_size || chan >= out_channels) return;

    float sum = 0.0f;
    int pooled_h_start, pooled_h_end, pooled_w_start, pooled_w_end;

    // Loop over output spatial locations for conv_transpose
    for (int h = 0; h < out_height; ++h) {
        for (int w = 0; w < out_width; ++w) {
            float val = 0.0f;

            // ConvTranspose2d: apply each kernel
            for (int k_c = 0; k_c < in_channels; ++k_c) {
                for (int k_h = 0; k_h < kernel_size; ++k_h) {
                    for (int k_w = 0; k_w < kernel_size; ++k_w) {
                        int in_h = h * stride - padding + k_h;
                        int in_w = w * stride - padding + k_w;

                        if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                            int in_idx = batch * in_channels * input_height * input_width +
                                        k_c * input_height * input_width +
                                        in_h * input_width + in_w;
                            int weight_idx = chan * in_channels * kernel_size * kernel_size +
                                            k_c * kernel_size * kernel_size +
                                            k_h * kernel_size + k_w;
                            val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            val += bias[chan];

            // Hardtanh activation
            val = fmaxf(hardtanh_min, fminf(hardtanh_max, val));

            // MaxPool: check if (h,w) is a starting point of a pooling window
            int pool_h = h / pool_stride;
            int pool_w = w / pool_stride;
            int pool_h_start = pool_h * pool_stride;
            int pool_w_start = pool_w * pool_stride;

            if (h == pool_h_start && w == pool_w_start) {
                float max_val = -FLT_MAX;
                for (int ph = 0; ph < pool_kernel_size; ++ph) {
                    for (int pw = 0; pw < pool_kernel_size; ++pw) {
                        int nh = h + ph;
                        int nw = w + pw;
                        if (nh < out_height && nw < out_width) {
                            // Recompute activation at (nh, nw) for pooling
                            float pool_val = 0.0f;
                            for (int k_c = 0; k_c < in_channels; ++k_c) {
                                for (int k_h = 0; k_h < kernel_size; ++k_h) {
                                    for (int k_w = 0; k_w < kernel_size; ++k_w) {
                                        int in_h = nh * stride - padding + k_h;
                                        int in_w = nw * stride - padding + k_w;
                                        if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                                            int in_idx = batch * in_channels * input_height * input_width +
                                                        k_c * input_height * input_width +
                                                        in_h * input_width + in_w;
                                            int weight_idx = chan * in_channels * kernel_size * kernel_size +
                                                            k_c * kernel_size * kernel_size +
                                                            k_h * kernel_size + k_w;
                                            pool_val += input[in_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                            pool_val += bias[chan];
                            pool_val = fmaxf(hardtanh_min, fminf(hardtanh_max, pool_val));
                            max_val = fmaxf(max_val, pool_val);
                        }
                    }
                }
                sum += max_val;
            }
        }
    }

    // Mean over spatial dimensions: divide by number of pooled output locations
    int pooled_h = (out_height + pool_stride - 1) / pool_stride;
    int pooled_w = (out_width + pool_stride - 1) / pool_stride;
    float mean_val = sum / (pooled_h * pooled_w);

    // Final tanh
    mean_val = tanhf(mean_val);

    // Output is (batch, out_channels, 1, 1)
    int out_idx = batch * out_channels + chan;
    output[out_idx] = mean_val;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int input_height, int input_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding, int output_padding,
    int pool_kernel_size, int pool_stride,
    float hardtanh_min, float hardtanh_max
);
"""

# Compile the fused CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for:
    ConvTranspose2d -> MaxPool2d -> Hardtanh -> Mean -> Tanh
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # Define transposed convolution parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

        # Store fused CUDA op
        self.fused_op = fused_op

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape

        # Output size of ConvTranspose2d
        out_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size

        # Launch fused kernel
        output = self.fused_op.fused_op_cuda(
            x, self.weight, self.bias,
            self.in_channels, input_height, input_width,
            self.out_channels, out_height, out_width,
            self.kernel_size, self.stride, self.padding, 0,  # output_padding=0
            self.maxpool_kernel_size, self.maxpool_stride,
            self.hardtanh_min, self.hardtanh_max
        )

        # Reshape to (batch, out_channels, 1, 1)
        output = output.view(batch_size, self.out_channels, 1, 1)
        return output