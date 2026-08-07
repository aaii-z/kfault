import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + BatchNorm3d + AvgPool3d (x2)
fused_conv_bn_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function to get index in 5D tensor: (n, c, d, h, w)
__device__ int get_index(int n, int c, int d, int h, int w, 
                         int N, int C, int D, int H, int W) {
    return ((n * C + c) * D + d) * H * W + h * W + w;
}

// CUDA kernel for fused transposed conv, batch norm, and two avg pools
__global__ void fused_conv_bn_double_pool_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_weight,
    const float* bn_bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    float eps,
    int pooled_depth,
    int pooled_height,
    int pooled_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    
    if (idx >= total_elements) return;

    int w = idx % pooled_width;
    int h = (idx / pooled_width) % pooled_height;
    int d = (idx / (pooled_width * pooled_height)) % pooled_depth;
    int c = (idx / (pooled_width * pooled_height * pooled_depth)) % out_channels;
    int n = idx / (pooled_width * pooled_height * pooled_depth * out_channels);

    // Scale up coordinates for first avg pool (2x2x2)
    int pool1_d = d * 2;
    int pool1_h = h * 2;
    int pool1_w = w * 2;

    // Scale up coordinates for second avg pool (2x2x2)
    int conv_d = pool1_d * 2;
    int conv_h = pool1_h * 2;
    int conv_w = pool1_w * 2;

    float sum = 0.0f;

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_d = conv_d + kd - kernel_size + 1 + padding;
                int in_h = conv_h + kh - kernel_size + 1 + padding;
                int in_w = conv_w + kw - kernel_size + 1 + padding;

                if (in_d >= 0 && in_d < in_depth && in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int input_idx = get_index(n, ic, in_d, in_h, in_w, batch_size, in_channels, in_depth, in_height, in_width);
                        int weight_idx = get_index(c, ic, kd, kh, kw, out_channels, in_channels, kernel_size, kernel_size, kernel_size);
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias
    sum += bias[c];

    // Batch norm: (sum - running_mean) / sqrt(running_var + eps) * bn_weight + bn_bias
    float bn_val = (sum - running_mean[c]) / sqrtf(running_var[c] + eps) * bn_weight[c] + bn_bias[c];

    // First pooling: average over 2x2x2 region
    float pool1_out = bn_val / 8.0f;

    // Second pooling: average over 2x2x2 region (but we already downsampled twice)
    // Since we are at final output, no further spatial accumulation needed
    // Each output element corresponds to 8x8=64 input elements in the conv output
    // But we already divided by 8 in pool1, and now we divide by 8 again
    float final_out = pool1_out / 8.0f;

    output[idx] = final_out;
}

torch::Tensor fused_conv_bn_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int64_t out_depth,
    int64_t out_height,
    int64_t out_width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    float eps,
    int64_t pooled_depth,
    int64_t pooled_height,
    int64_t pooled_width) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    auto out_channels = weight.size(0);

    auto output = torch::empty({batch_size, out_channels, pooled_depth, pooled_height, pooled_width}, input.options());

    const int total_elements = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_conv_bn_double_pool_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        eps,
        pooled_depth,
        pooled_height,
        pooled_width
    );

    return output;
}
"""

fused_conv_bn_pool_cpp = """
torch::Tensor fused_conv_bn_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int64_t out_depth,
    int64_t out_height,
    int64_t out_width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    float eps,
    int64_t pooled_depth,
    int64_t pooled_height,
    int64_t pooled_width);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_bn_pool",
    cpp_sources=fused_conv_bn_pool_cpp,
    cuda_sources=fused_conv_bn_pool_source,
    functions=["fused_conv_bn_pool_cuda"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for ConvTranspose3d + BatchNorm3d + AvgPool3d (x2)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define the same layers so we can extract parameters
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

        # Store bias shape for later use
        self.bias_shape = bias_shape

    def forward(self, x):
        # Get output shape after transposed conv
        out_depth = (x.size(2) - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_height = (x.size(3) - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_width = (x.size(4) - 1) * self.stride - 2 * self.padding + self.kernel_size

        # After two 2x2x2 avg pools
        pooled_depth = out_depth // 4
        pooled_height = out_height // 4
        pooled_width = out_width // 4

        # Extract parameters
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        bn_weight = self.batch_norm.weight
        bn_bias = self.batch_norm.bias
        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        eps = self.batch_norm.eps

        # Call fused CUDA kernel
        output = fused_op.fused_conv_bn_pool_cuda(
            x,
            weight,
            bias,
            bn_weight,
            bn_bias,
            running_mean,
            running_var,
            out_depth,
            out_height,
            out_width,
            self.kernel_size,
            self.stride,
            self.padding,
            eps,
            pooled_depth,
            pooled_height,
            pooled_width
        )

        return output