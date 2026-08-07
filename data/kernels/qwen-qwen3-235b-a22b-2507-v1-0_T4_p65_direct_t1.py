import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + AvgPool2d + Sigmoid + Sum
fused_conv_avgpool_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function to calculate 2D output size after convolution
__device__ int conv_output_size(int input_size, int kernel_size, int stride, int padding) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

// Helper function to calculate 2D output size after avg pooling
__device__ int pool_output_size(int input_size, int kernel_size, int stride, int padding) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

// CUDA kernel for fused Conv2d + AvgPool2d + Sigmoid + Sum
__global__ void fused_conv_avgpool_sigmoid_sum_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    // Input tensor shape
    int batch_size, int in_channels, int in_height, int in_width,
    // Conv params
    int conv_kernel_h, int conv_kernel_w,
    int conv_stride_h, int conv_stride_w,
    int conv_padding_h, int conv_padding_w,
    int out_channels,
    // AvgPool params
    int pool_kernel, int pool_stride
) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;

    float sum = 0.0f;

    // Output spatial dimensions after conv and pooling
    int conv_out_h = conv_output_size(in_height, conv_kernel_h, conv_stride_h, conv_padding_h);
    int conv_out_w = conv_output_size(in_width, conv_kernel_w, conv_stride_w, conv_padding_w);
    int pooled_h = pool_output_size(conv_out_h, pool_kernel, pool_stride, 0);
    int pooled_w = pool_output_size(conv_out_w, pool_kernel, pool_stride, 0);

    // Loop over output channels, pooled height and width
    for (int oc = 0; oc < out_channels; oc++) {
        for (int ph = 0; ph < pooled_h; ph++) {
            for (int pw = 0; pw < pooled_w; pw++) {
                float conv_value = 0.0f;

                // Starting position in conv output for pooling
                int h_start = ph * pool_stride;
                int w_start = pw * pool_stride;

                // Average pool window: sum over pool_kernel x pool_kernel
                float pool_sum = 0.0f;
                int pool_count = 0;

                for (int kh = 0; kh < pool_kernel; kh++) {
                    for (int kw = 0; kw < pool_kernel; kw++) {
                        int h_pos = h_start + kh;
                        int w_pos = w_start + kw;

                        // Check bounds in conv output
                        if (h_pos >= 0 && h_pos < conv_out_h && w_pos >= 0 && w_pos < conv_out_w) {
                            float val = 0.0f;

                            // Perform convolution at (h_pos, w_pos)
                            for (int ic = 0; ic < in_channels; ic++) {
                                for (int ckh = 0; ckh < conv_kernel_h; ckh++) {
                                    for (int ckw = 0; ckw < conv_kernel_w; ckw++) {
                                        int ih = h_pos * conv_stride_h + ckh - conv_padding_h;
                                        int iw = w_pos * conv_stride_w + ckw - conv_padding_w;

                                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                            int input_idx = batch * in_channels * in_height * in_width +
                                                            ic * in_height * in_width +
                                                            ih * in_width + iw;
                                            int weight_idx = oc * in_channels * conv_kernel_h * conv_kernel_w +
                                                             ic * conv_kernel_h * conv_kernel_w +
                                                             ckh * conv_kernel_w + ckw;
                                            val += input[input_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                            val += bias[oc];  // Add bias
                            pool_sum += val;
                            pool_count++;
                        }
                    }
                }

                float avg_pooled = pool_count > 0 ? pool_sum / pool_count : 0.0f;
                float sigmoid_out = 1.0f / (1.0f + expf(-avg_pooled));
                sum += sigmoid_out;
            }
        }
    }

    output[batch] = sum;
}

// C++ wrapper function
torch::Tensor fused_conv_avgpool_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_height, int in_width,
    int conv_kernel_h, int conv_kernel_w,
    int conv_stride_h, int conv_stride_w,
    int conv_padding_h, int conv_padding_w,
    int out_channels,
    int pool_kernel, int pool_stride
) {
    auto batch_size = input.size(0);
    auto output = torch::zeros({batch_size}, input.options());

    const int block_size = 256;
    const int num_blocks = batch_size;

    fused_conv_avgpool_sigmoid_sum_kernel<<<num_blocks, 1>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, input.size(1), in_height, in_width,
        conv_kernel_h, conv_kernel_w,
        conv_stride_h, conv_stride_w,
        conv_padding_h, conv_padding_w,
        out_channels,
        pool_kernel, pool_stride
    );

    return output;
}
"""

fused_conv_avgpool_sigmoid_sum_cpp = """
torch::Tensor fused_conv_avgpool_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_height, int in_width,
    int conv_kernel_h, int conv_kernel_w,
    int conv_stride_h, int conv_stride_w,
    int conv_padding_h, int conv_padding_w,
    int out_channels,
    int pool_kernel, int pool_stride);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_avgpool_sigmoid_sum",
    cpp_sources=fused_conv_avgpool_sigmoid_sum_cpp,
    cuda_sources=fused_conv_avgpool_sigmoid_sum_source,
    functions=["fused_conv_avgpool_sigmoid_sum_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + AvgPool2d + Sigmoid + Sum.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

        # Create the convolution layer to reuse weight and bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

        # Conv parameters
        self.conv_stride = self.conv.stride
        self.conv_padding = self.conv.padding

        # Pooling parameters
        self.pool_kernel = pool_kernel_size
        self.pool_stride = pool_kernel_size  # Assuming no stride override

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape

        # Launch fused CUDA kernel
        output = fused_op.fused_conv_avgpool_sigmoid_sum_cuda(
            x,
            self.weight,
            self.bias,
            in_height, in_width,
            self.kernel_size, self.kernel_size,
            self.conv_stride[0], self.conv_stride[1],
            self.conv_padding[0], self.conv_padding[1],
            self.out_channels,
            self.pool_kernel, self.pool_stride
        )
        return output