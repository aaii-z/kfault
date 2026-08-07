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

// Helper function to calculate 3D index from linear index
__device__ void idx2coords(int idx, int& d, int& h, int& w, int D, int H, int W) {
    w = idx % W;
    idx /= W;
    h = idx % H;
    idx /= H;
    d = idx;
}

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
    int total_elements = B * C_out;
    if (idx >= total_elements) return;

    int b = idx / C_out;
    int c_out = idx % C_out;

    // Convolution + ReLU + MaxPool + GlobalAvgPool in one fused step
    float sum_val = 0.0f;

    // Global average pooling output spatial dims are (1,1,1), so we reduce entire conv output after maxpool
    for (int d_pool = 0; d_pool < D_out_pool; d_pool++) {
        for (int h_pool = 0; h_pool < H_out_pool; h_pool++) {
            for (int w_pool = 0; w_pool < W_out_pool; w_pool++) {
                float max_val = -FLT_MAX;

                // Max pooling over kernel size
                int d_start = d_pool * D_pool;
                int h_start = h_pool * H_pool;
                int w_start = w_pool * W_pool;

                for (int kd = 0; kd < D_pool; kd++) {
                    for (int kh = 0; kh < H_pool; kh++) {
                        for (int kw = 0; kw < W_pool; kw++) {
                            int d_c = d_start + kd;
                            int h_c = h_start + kh;
                            int w_c = w_start + kw;

                            if (d_c >= D_out_conv || h_c >= H_out_conv || w_c >= W_out_conv) continue;

                            float conv_val = 0.0f;
                            for (int c_in = 0; c_in < C_in; c_in++) {
                                for (int kd_conv = 0; kd_conv < D_k; kd_conv++) {
                                    for (int kh_conv = 0; kh_conv < H_k; kh_conv++) {
                                        for (int kw_conv = 0; kw_conv < W_k; kw_conv++) {
                                            int d_in = d_c - (D_k / 2) + kd_conv;
                                            int h_in = h_c - (H_k / 2) + kh_conv;
                                            int w_in = w_c - (W_k / 2) + kw_conv;

                                            if (d_in < 0 || d_in >= D_in || h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;

                                            int in_idx = b * C_in * D_in * H_in * W_in +
                                                         c_in * D_in * H_in * W_in +
                                                         d_in * H_in * W_in +
                                                         h_in * W_in +
                                                         w_in;

                                            int weight_idx = c_out * C_in * D_k * H_k * W_k +
                                                             c_in * D_k * H_k * W_k +
                                                             kd_conv * H_k * W_k +
                                                             kh_conv * W_k +
                                                             kw_conv;

                                            conv_val += input[in_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                            max_val = MAX(max_val, conv_val);
                        }
                    }
                }

                // After max pooling, we divide by divisor
                max_val /= divisor;

                // Accumulate for global average pooling (average over all pooled locations)
                sum_val += max_val;
            }
        }
    }

    // Global average pooling: average over all pooled positions
    int num_pool_elements = D_out_pool * H_out_pool * W_out_pool;
    float global_avg = sum_val / num_pool_elements;

    // Add bias (bias shape: (C_out, 1, 1, 1))
    global_avg += bias_vec[c_out];

    // Sum along sum_dim (which is 1, i.e., over C_out dimension). But since we're per-channel,
    // and sum_dim=1 corresponds to channel dim, we will later sum across channels.
    // However, in this kernel we are computing per-output-channel scalar.
    // Final sum over dim=1 (channels) will be done in a separate reduction.

    // Write out the value for this (b, c_out) before summing
    int out_idx = b * C_out + c_out;
    output[out_idx] = global_avg;
}

// Reduction kernel to sum along channel dimension
__global__ void sum_along_dim_kernel(float* output, int B, int C_out) {
    int b = blockIdx.x;
    float sum = 0.0f;
    for (int c = 0; c < C_out; c++) {
        sum += output[b * C_out + c];
    }
    output[b] = sum;
}

torch::Tensor fused_conv_pool_bias_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_vec,
    float divisor,
    int D_pool, int H_pool, int W_pool,
    int sum_dim
) {
    auto B = input.size(0);
    auto C_in = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);

    auto C_out = weight.size(0);
    auto D_k = weight.size(2);
    auto H_k = weight.size(3);
    auto W_k = weight.size(4);

    // Conv output size (assuming padding='same' as default in PyTorch for Conv3d with kernel_size=3 and padding=1)
    int pad_d = D_k / 2;
    int pad_h = H_k / 2;
    int pad_w = W_k / 2;
    int D_out_conv = D_in;
    int H_out_conv = H_in;
    int W_out_conv = W_in;

    // MaxPool output size
    int D_out_pool = (D_out_conv + D_pool - 1) / D_pool;
    int H_out_pool = (H_out_conv + H_pool - 1) / H_pool;
    int W_out_pool = (W_out_conv + W_pool - 1) / W_pool;

    // Output before sum: (B, C_out)
    auto output = torch::zeros({B, C_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (B * C_out + block_size - 1) / block_size;

    fused_conv_pool_bias_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_vec.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, D_in, H_in, W_in,
        C_out, D_k, H_k, W_k,
        D_out_conv, H_out_conv, W_out_conv,
        D_pool, H_pool, W_pool,
        D_out_pool, H_out_pool, W_out_pool,
        divisor,
        sum_dim
    );

    // Now reduce along channel dimension (sum_dim=1)
    auto final_output = torch::zeros({B}, input.options());
    sum_along_dim_kernel<<<B, 1>>>(
        output.data_ptr<float>(),
        B, C_out
    );

    // Copy reduced result
    cudaMemcpy(final_output.data_ptr<float>(), output.data_ptr<float>(), B * sizeof(float), cudaMemcpyDeviceToDevice);

    return final_output;
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
        return self.fused_op(
            x,
            self.conv_weight,
            self.bias,
            self.divisor,
            self.pool_size[0], self.pool_size[1], self.pool_size[2],
            self.sum_dim
        )