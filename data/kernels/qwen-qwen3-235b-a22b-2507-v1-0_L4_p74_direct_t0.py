import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + LeakyReLU + Multiply + LeakyReLU + MaxPool3d
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Helper function to calculate 5D index from linear index
__device__ inline int get_index(int d, int h, int w, int D, int H, int W) {
    int di = d;
    int hi = h;
    int wi = w;
    return di * H * W + hi * W + wi;
}

__global__ void fused_convtrans_leakyrelu_mul_leakyrelu_maxpool(
    const float* input,
    const float* weight,
    const float* bias,
    const float* multiplier,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions
    int out_batch, int out_channels, int out_depth, int out_height, int out_width,
    // Conv transpose params
    int kernel_size, int stride, int padding, int output_padding,
    // Weight dimensions
    int w_depth, int w_height, int w_width,
    // Maxpool params
    int pool_kernel, int pool_stride,
    // Output pooled dimensions
    int pooled_depth, int pooled_height, int pooled_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = out_batch * out_channels * pooled_depth * pooled_height * pooled_width;
    
    if (idx >= total_elements) return;

    // Decode output pooled index
    int wo = idx;
    int w_pooled = wo % pooled_width; wo /= pooled_width;
    int h_pooled = wo % pooled_height; wo /= pooled_height;
    int d_pooled = wo % pooled_depth; wo /= pooled_depth;
    int c = wo % out_channels;
    int b = wo / out_channels;

    // Map pooled indices to input indices of activation (before pooling)
    int d_start = d_pooled * pool_stride;
    int h_start = h_pooled * pool_stride;
    int w_start = w_pooled * pool_stride;
    int d_end = d_start + pool_kernel;
    int h_end = h_start + pool_kernel;
    int w_end = w_start + pool_kernel;

    float max_val = -FLT_MAX;

    // Loop over pooling window
    for (int dk = d_start; dk < d_end; ++dk) {
        for (int hk = h_start; hk < h_end; ++hk) {
            for (int wk = w_start; wk < w_end; ++wk) {
                // Skip if outside the activation map
                if (dk < 0 || dk >= out_depth || hk < 0 || hk >= out_height || wk < 0 || wk >= out_width) continue;

                // Compute conv transpose output at (dk, hk, wk)
                float conv_val = 0.0f;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kz = 0; kz < w_depth; ++kz) {
                        for (int ky = 0; ky < w_height; ++ky) {
                            for (int kx = 0; kx < w_width; ++kx) {
                                int di = dk * stride - 2 * padding + kz;
                                int hi = hk * stride - 2 * padding + ky;
                                int wi = wk * stride - 2 * padding + kx;

                                if (di < 0 || di >= in_depth || hi < 0 || hi >= in_height || wi < 0 || wi >= in_width) continue;

                                int in_idx = b * in_channels * in_depth * in_height * in_width +
                                           ic * in_depth * in_height * in_width +
                                           di * in_height * in_width +
                                           hi * in_width +
                                           wi;

                                int weight_idx = c * in_channels * w_depth * w_height * w_width +
                                               ic * w_depth * w_height * w_width +
                                               kz * w_height * w_width +
                                               ky * w_width +
                                               kx;

                                conv_val += input[in_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                conv_val += bias[c];

                // First LeakyReLU
                float lr1 = (conv_val > 0) ? conv_val : 0.2f * conv_val;

                // Multiply by channel-wise multiplier
                float mul_val = lr1 * multiplier[c];

                // Second LeakyReLU
                float lr2 = (mul_val > 0) ? mul_val : 0.2f * mul_val;

                max_val = MAX(max_val, lr2);
            }
        }
    }

    // Write max-pooled result
    int out_idx = b * out_channels * pooled_depth * pooled_height * pooled_width +
                  c * pooled_depth * pooled_height * pooled_width +
                  d_pooled * pooled_height * pooled_width +
                  h_pooled * pooled_width +
                  w_pooled;
    output[out_idx] = max_val;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    int in_depth, int in_height, int in_width,
    int kernel_size, int stride, int padding, int output_padding,
    int pool_kernel, int pool_stride
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        # ConvTranspose3d parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Initialize weights and bias for ConvTranspose3d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

        # Learnable multiplier
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        # MaxPool3d parameters
        self.pool_kernel = 2
        self.pool_stride = 2

        # Store the fused CUDA function
        self.fused_op = fused_op.fused_op_cuda

    def forward(self, x):
        # Calculate output shape of ConvTranspose3d
        # Formula: (input - 1) * stride - 2 * padding + kernel_size + output_padding
        in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
        out_depth = (in_depth - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Output shape after max pooling (kernel=2, stride=2)
        pooled_depth = out_depth // 2
        pooled_height = out_height // 2
        pooled_width = out_width // 2

        # Call the fused CUDA kernel
        output = self.fused_op(
            x,
            self.weight,
            self.bias,
            self.multiplier,
            in_depth, in_height, in_width,
            self.kernel_size, self.stride, self.padding, self.output_padding,
            self.pool_kernel, self.pool_stride
        )

        # Reshape output to (batch, out_channels, pooled_depth, pooled_height, pooled_width)
        output = output.view(x.size(0), self.out_channels, pooled_depth, pooled_height, pooled_width)
        return output