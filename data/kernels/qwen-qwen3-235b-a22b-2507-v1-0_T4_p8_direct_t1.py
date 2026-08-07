import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused 3D convolution, division, max pooling, global average pooling, bias add, and sum
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Fused kernel: Conv3D -> Divide -> MaxPool3D -> GlobalAvgPool3D -> BiasAdd -> Sum
__global__ void fused_conv_pool_bias_sum_kernel(
    const float* input,
    const float* weight,
    const float* bias_vec,
    float* output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_k, int H_k, int W_k,
    int D_out_conv, int H_out_conv, int W_out_conv,
    int D_pool, int H_pool, int W_pool,
    int D_out_pool, int H_out_pool, int W_out_pool,
    float divisor,
    int sum_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * 1 * 1 * 1;
    if (idx >= total_elements) return;

    int b = idx / (C_out * 1 * 1 * 1);
    int c_out = (idx / (1 * 1 * 1)) % C_out;

    float sum_val = 0.0f;

    // Max pooling loop
    for (int d_pool = 0; d_pool < D_out_pool; d_pool++) {
        for (int h_pool = 0; h_pool < H_out_pool; h_pool++) {
            for (int w_pool = 0; w_pool < W_out_pool; w_pool++) {
                float max_val = -FLT_MAX;

                // Max pooling over pool kernel
                int d_start = d_pool * D_pool;
                int h_start = h_pool * H_pool;
                int w_start = w_pool * W_pool;

                for (int pd = 0; pd < D_pool; pd++) {
                    for (int ph = 0; ph < H_pool; ph++) {
                        for (int pw = 0; pw < W_pool; pw++) {
                            int d_c = d_start + pd;
                            int h_c = h_start + ph;
                            int w_c = w_start + pw;

                            if (d_c >= D_out_conv || h_c >= H_out_conv || w_c >= W_out_conv) continue;

                            float conv_val = 0.0f;
                            for (int c_in = 0; c_in < C_in; c_in++) {
                                for (int kd = 0; kd < D_k; kd++) {
                                    for (int kh = 0; kh < H_k; kh++) {
                                        for (int kw = 0; kw < W_k; kw++) {
                                            int d_in = d_c - (D_k / 2) + kd;
                                            int h_in = h_c - (H_k / 2) + kh;
                                            int w_in = w_c - (W_k / 2) + kw;

                                            if (d_in < 0 || d_in >= D_in || h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;

                                            int in_idx = b * C_in * D_in * H_in * W_in +
                                                         c_in * D_in * H_in * W_in +
                                                         d_in * H_in * W_in +
                                                         h_in * W_in +
                                                         w_in;

                                            int weight_idx = c_out * C_in * D_k * H_k * W_k +
                                                             c_in * D_k * H_k * W_k +
                                                             kd * H_k * W_k +
                                                             kh * W_k +
                                                             kw;

                                            conv_val += input[in_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                            max_val = MAX(max_val, conv_val);
                        }
                    }
                }

                max_val /= divisor;
                sum_val += max_val;
            }
        }
    }

    // Global average pooling: average over pooled spatial dimensions
    int num_pool_elements = D_out_pool * H_out_pool * W_out_pool;
    float global_avg = sum_val / num_pool_elements;

    // Add bias (C_out, 1, 1, 1)
    global_avg += bias_vec[c_out];

    // Output shape: (B, C_out, 1, 1, 1)
    int out_idx = b * C_out * 1 * 1 * 1 + c_out * 1 * 1 * 1;
    output[out_idx] = global_avg;
}

// Reduction kernel to sum along channel dimension (dim=1)
__global__ void sum_along_dim_kernel(float* input_output, int B, int C_out) {
    int b = blockIdx.x;
    float sum = 0.0f;
    for (int c = 0; c < C_out; c++) {
        sum += input_output[b * C_out * 1 * 1 * 1 + c];
    }
    input_output[b] = sum;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_pool_bias_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_vec,
    float divisor,
    int D_pool, int H_pool, int W_pool,
    int sum_dim
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_pool_bias_sum",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_pool_bias_sum_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for 3D convolution, division, max pooling,
    global average pooling, bias addition, and sum along dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.conv_weight)
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.fused_op = fused_op.fused_conv_pool_bias_sum_cuda

    def forward(self, x):
        # Run fused kernel: outputs (B, C_out, 1, 1, 1)
        y = self.fused_op(
            x,
            self.conv_weight,
            self.bias,
            self.divisor,
            self.pool_size[0], self.pool_size[1], self.pool_size[2],
            self.sum_dim
        )

        # Now sum along channel dimension (dim=1)
        z = torch.sum(y, dim=self.sum_dim, keepdim=True)  # Keep dim to match shape (B, 1, 1, 1, 1)
        return z.squeeze(1)  # Remove channel dim -> (B, 1, 1, 1)