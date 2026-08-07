import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution, group normalization, scaling, max pooling, and clamping
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Define constants for kernel sizes
#define CONV_KERNEL_SIZE 3
#define MAXPOOL_KERNEL_SIZE 4
#define BLOCK_SIZE 16

// Convolution kernel
__global__ void conv2d_kernel(const float* input, const float* weight, const float* bias, float* output,
                              int batch_size, int in_channels, int out_channels, int height, int width) {
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = blockIdx.z % out_channels;
    int b = blockIdx.z / out_channels;

    if (out_h < height && out_w < width) {
        float sum = bias[out_c];
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int kh = 0; kh < CONV_KERNEL_SIZE; ++kh) {
                for (int kw = 0; kw < CONV_KERNEL_SIZE; ++kw) {
                    int in_h = out_h + kh - 1;
                    int in_w = out_w + kw - 1;
                    if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        int input_idx = b * (in_channels * height * width) + in_c * (height * width) + in_h * width + in_w;
                        int weight_idx = out_c * (in_channels * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE) + in_c * (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE) + kh * CONV_KERNEL_SIZE + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        int output_idx = b * (out_channels * height * width) + out_c * (height * width) + out_h * width + out_w;
        output[output_idx] = sum;
    }
}

// Group Normalization + Scaling kernel
__global__ void group_norm_scale_kernel(float* data, const float* scale, int batch_size, int channels, int height, int width, int num_groups) {
    int b = blockIdx.x;
    int g = blockIdx.y;
    int c_start = g * (channels / num_groups);
    int c_end = (g + 1) * (channels / num_groups);
    int spatial_size = height * width;
    int group_size = (channels / num_groups) * spatial_size;

    if (b < batch_size && g < num_groups) {
        // Compute mean and variance
        float mean = 0.0f;
        float var = 0.0f;
        for (int c = c_start; c < c_end; ++c) {
            for (int i = 0; i < spatial_size; ++i) {
                int idx = b * (channels * spatial_size) + c * spatial_size + i;
                mean += data[idx];
            }
        }
        mean /= group_size;

        for (int c = c_start; c < c_end; ++c) {
            for (int i = 0; i < spatial_size; ++i) {
                int idx = b * (channels * spatial_size) + c * spatial_size + i;
                float diff = data[idx] - mean;
                var += diff * diff;
            }
        }
        var /= group_size;

        // Normalize and scale
        float inv_std = rsqrtf(var + 1e-5f);
        for (int c = c_start; c < c_end; ++c) {
            float s = scale[c];
            for (int i = 0; i < spatial_size; ++i) {
                int idx = b * (channels * spatial_size) + c * spatial_size + i;
                data[idx] = (data[idx] - mean) * inv_std * s;
            }
        }
    }
}

// Max Pooling + Clamping kernel
__global__ void maxpool_clamp_kernel(const float* input, float* output, int batch_size, int channels, int height, int width, float clamp_min, float clamp_max) {
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    int out_height = (height - MAXPOOL_KERNEL_SIZE) / MAXPOOL_KERNEL_SIZE + 1;
    int out_width = (width - MAXPOOL_KERNEL_SIZE) / MAXPOOL_KERNEL_SIZE + 1;

    if (out_h < out_height && out_w < out_width) {
        float max_val = -FLT_MAX;
        for (int kh = 0; kh < MAXPOOL_KERNEL_SIZE; ++kh) {
            for (int kw = 0; kw < MAXPOOL_KERNEL_SIZE; ++kw) {
                int in_h = out_h * MAXPOOL_KERNEL_SIZE + kh;
                int in_w = out_w * MAXPOOL_KERNEL_SIZE + kw;
                int input_idx = b * (channels * height * width) + c * (height * width) + in_h * width + in_w;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
        max_val = fmaxf(clamp_min, fminf(clamp_max, max_val));
        int output_idx = b * (channels * out_height * out_width) + c * (out_height * out_width) + out_h * out_width + out_w;
        output[output_idx] = max_val;
    }
}

// Helper function to launch kernels
torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor conv_weight, torch::Tensor conv_bias, torch::Tensor scale, int num_groups, float clamp_min, float clamp_max) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = conv_weight.size(0);

    // Allocate temporary buffer for conv output
    auto conv_output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    // Launch conv2d kernel
    dim3 conv_grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size * out_channels);
    dim3 conv_block(BLOCK_SIZE, BLOCK_SIZE);
    conv2d_kernel<<<conv_grid, conv_block>>>(input.data_ptr<float>(), conv_weight.data_ptr<float>(), conv_bias.data_ptr<float>(), conv_output.data_ptr<float>(),
                                             batch_size, in_channels, out_channels, height, width);

    // Launch group norm + scale kernel
    dim3 gn_grid(batch_size, num_groups);
    group_norm_scale_kernel<<<gn_grid, 1>>>(conv_output.data_ptr<float>(), scale.data_ptr<float>(), batch_size, out_channels, height, width, num_groups);

    // Allocate output tensor
    int out_height = (height - MAXPOOL_KERNEL_SIZE) / MAXPOOL_KERNEL_SIZE + 1;
    int out_width = (width - MAXPOOL_KERNEL_SIZE) / MAXPOOL_KERNEL_SIZE + 1;
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Launch maxpool + clamp kernel
    dim3 mp_grid((out_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size * out_channels);
    maxpool_clamp_kernel<<<mp_grid, conv_block>>>(conv_output.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_channels, height, width, clamp_min, clamp_max);

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor conv_weight, torch::Tensor conv_bias, torch::Tensor scale, int num_groups, float clamp_min, float clamp_max);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that uses custom CUDA kernels for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Initialize parameters that will be passed to CUDA kernels
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.num_groups = num_groups
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(x, self.conv_weight, self.conv_bias, self.scale, self.num_groups, self.clamp_min, self.clamp_max)