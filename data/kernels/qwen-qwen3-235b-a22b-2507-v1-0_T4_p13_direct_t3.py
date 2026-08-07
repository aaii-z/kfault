import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused ConvTranspose3d + Mean Pooling + Bias Add + Softmax + Tanh + Scaling
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Calculate 5D tensor index: (b, c, d, h, w)
__device__ int index_5d(int b, int c, int d, int h, int w, int C, int D, int H, int W) {
    return ((b * C + c) * D + d) * H * W + h * W + w;
}

__global__ void fused_conv_transpose_mean_softmax_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias_vec,
    float* output,
    int B, int IN_C, int OUT_C, int D, int H, int W,
    int K, int stride, int padding,
    float scaling_factor,
    int OUT_D, int OUT_H, int OUT_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OUT_C * OUT_H * OUT_W;
    if (idx >= total_elements) return;

    int w = idx % OUT_W;
    int h = (idx / OUT_W) % OUT_H;
    int c = (idx / (OUT_H * OUT_W)) % OUT_C;
    int b = idx / (OUT_C * OUT_H * OUT_W);

    // ConvTranspose3d + mean over depth
    float sum_val = 0.0f;
    for (int ic = 0; ic < IN_C; ++ic) {
        for (int ki = 0; ki < K; ++ki) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int i_d = ki;  // kernel iterates over output depth
                    int i_h = h * stride - padding + kh;
                    int i_w = w * stride - padding + kw;

                    if (i_h >= 0 && i_h < H && i_w >= 0 && i_w < W) {
                        float w_val = weight[c * IN_C * K * K * K + ic * K * K * K + ki * K * K + kh * K + kw];
                        float input_sum = 0.0f;
                        for (int id = 0; id < D; ++id) {
                            int in_idx = index_5d(b, ic, id, i_h, i_w, IN_C, D, H, W);
                            input_sum += input[in_idx];
                        }
                        input_sum /= D;  // mean over depth
                        sum_val += w_val * input_sum;
                    }
                }
            }
        }
    }

    // Add bias
    sum_val += bias_vec[c];

    // Each spatial location (h, w) has OUT_C channels; use shared memory for softmax per spatial location
    int spatial_id = h * OUT_W + w;
    extern __shared__ float shared[];
    float* shared_softmax = shared + spatial_id * OUT_C;
    shared_softmax[c] = sum_val;
    __syncthreads();

    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int cc = 0; cc < OUT_C; ++cc) {
        max_val = fmaxf(max_val, shared_softmax[cc]);
    }
    __syncthreads();

    // Compute exp(sum_val - max) and sum over channels
    float exp_val = __expf(sum_val - max_val);
    float exp_sum = 0.0f;
    for (int cc = 0; cc < OUT_C; ++cc) {
        exp_sum += __expf(shared_softmax[cc] - max_val);
    }
    __syncthreads();

    // Softmax, tanh, and scale
    float softmax_out = exp_val / (exp_sum + 1e-10f);
    float result = tanhf(softmax_out) * scaling_factor;

    // Write to output: (B, OUT_C, 1, OUT_H, OUT_W) -> we pool depth to 1, so output depth is 1
    int out_idx = b * OUT_C * OUT_H * OUT_W + c * OUT_H * OUT_W + h * OUT_W + w;
    output[out_idx] = result;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_vec,
    float scaling_factor,
    int stride,
    int padding,
    int K,
    int OUT_D,
    int OUT_H,
    int OUT_W);
"""

# Note: OUT_D is fixed to 1 due to mean(dim=2, keepdim=True)
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scaling_factor = scaling_factor

        # Transposed convolution weights: (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        # Bias is now 1D: (out_channels,)
        self.bias = nn.Parameter(torch.randn(out_channels))

        # Output dimensions (assuming input spatial size leads to these outputs)
        self.OUT_D = 1   # due to mean over depth
        self.OUT_H = 128
        self.OUT_W = 128

        self.fused_op = fused_op.fused_op_cuda

    def forward(self, x):
        # Launch kernel with shared memory size = number of spatial locations * OUT_C
        total_spatial = self.OUT_H * self.OUT_W
        shared_mem_size = total_spatial * self.out_channels * sizeof(float)

        return self.fused_op(
            x,
            self.weight,
            self.bias,
            self.scaling_factor,
            self.stride,
            self.padding,
            self.kernel_size,
            self.OUT_D,
            self.OUT_H,
            self.OUT_W
        )