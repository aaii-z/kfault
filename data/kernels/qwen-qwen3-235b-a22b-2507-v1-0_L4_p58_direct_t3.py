import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused transposed convolution, logsumexp, HardSwish, bias subtraction, and clamp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float fast_sigmoid(float x) {
    return 0.5f + 0.5f * x / (1.0f + fabsf(x));
}

__global__ void fused_transpose_conv_lse_hswish_clamp_kernel(
    const float* input,
    const float* weight,
    const float* bias_param,
    float* output,
    int B, int C_in, int D, int H, int W,
    int C_out, int D_out, int H_out, int W_out,
    int k, int stride, int pad
) {
    int batch_spatial_idx = blockIdx.x;
    int c_out = threadIdx.x;

    int total_output_elements_per_batch = D_out * H_out * W_out;
    int b = batch_spatial_idx / total_output_elements_per_batch;
    int d_out = (batch_spatial_idx % total_output_elements_per_batch) / (H_out * W_out);
    int h_out = ((batch_spatial_idx % total_output_elements_per_batch) % (H_out * W_out)) / W_out;
    int w_out = ((batch_spatial_idx % total_output_elements_per_batch) % (H_out * W_out)) % W_out;

    float conv_val = 0.0f;
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kd = 0; kd < k; kd++) {
            for (int kh = 0; kh < k; kh++) {
                for (int kw = 0; kw < k; kw++) {
                    int d_in = d_out * stride - pad + kd;
                    int h_in = h_out * stride - pad + kh;
                    int w_in = w_out * stride - pad + kw;

                    if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int in_idx = b * C_in * D * H * W + c_in * D * H * W + d_in * H * W + h_in * W + w_in;
                        int w_idx = c_out * C_in * k * k * k + c_in * k * k * k + kd * k * k + kh * k + kw;
                        conv_val += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    extern __shared__ float shared_data[];
    float* values = shared_data;
    values[c_out] = conv_val;
    __syncthreads();

    if (c_out == 0) {
        float max_val = -1e30f;
        for (int c = 0; c < C_out; c++) {
            max_val = fmaxf(max_val, values[c]);
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < C_out; c++) {
            sum_exp += expf(values[c] - max_val);
        }
        float lse_val = max_val + logf(sum_exp);
        float hswish_val = lse_val * fast_sigmoid(lse_val + 3.0f) / 6.0f;
        float result = fmaxf(-1.0f, fminf(1.0f, hswish_val - bias_param[0]));
        output[batch_spatial_idx] = result;
    }
}
"""

fused_cpp_source = """
torch::Tensor fused_transpose_conv_lse_hswish_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_param,
    int C_in, int C_out, int D, int H, int W,
    int k, int stride, int pad
);
"""

fused_op = load_inline(
    name="fused_transpose_conv_lse_hswish_clamp",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_transpose_conv_lse_hswish_clamp_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.fused_op = fused_op

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias_param = self.bias
        B, C_in, D, H, W = x.shape
        k = self.kernel_size
        stride = self.stride
        pad = self.padding
        C_out = self.out_channels

        D_out = (D - 1) * stride - 2 * pad + k
        H_out = (H - 1) * stride - 2 * pad + k
        W_out = (W - 1) * stride - 2 * pad + k

        total_output_elements = B * D_out * H_out * W_out
        shared_mem_size = C_out * sizeof(float)

        output = self.fused_op.fused_transpose_conv_lse_hswish_clamp_cuda(
            x, weight, bias_param,
            C_in, C_out, D, H, W,
            k, stride, pad
        )
        output = output.view(B, 1, D_out, H_out, W_out)
        return output