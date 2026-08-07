import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused ConvTranspose3d + Mean Pooling + Bias Add + Softmax + Tanh + Scaling
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Efficient fused kernel for:
// 1. Transposed 3D convolution (with implicit depth mean pooling)
// 2. Bias addition
// 3. Softmax over channels
// 4. Tanh activation
// 5. Scaling

// Utility function to calculate 5D tensor index: (b, c, d, h, w)
__device__ int get_index(int b, int c, int d, int h, int w, 
                         int C, int D, int H, int W) {
    return ((b * C + c) * D + d) * H * W + h * W + w;
}

__global__ void fused_conv_transpose_mean_softmax_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int IN_C, int OUT_C, int D, int H, int W,
    int K, int stride, int padding,
    float scaling_factor,
    int OUT_D, int OUT_H, int OUT_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OUT_C * OUT_H * OUT_W;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % OUT_W;
    int h = (idx / OUT_W) % OUT_H;
    int c = (idx / (OUT_H * OUT_W)) % OUT_C;
    int b = idx / (OUT_C * OUT_H * OUT_W);

    // ConvTranspose3d: sum over input channels and kernel space
    float sum_val = 0.0f;

    int d_start = 0;
    int d_end = D;

    // Iterate over kernel space
    for (int ki = 0; ki < K; ++ki) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int i_d = 0; // We will average over depth, so fix d=0 in output
                int i_h = h * stride - padding + kh;
                int i_w = w * stride - padding + kw;

                if (i_h >= 0 && i_h < H && i_w >= 0 && i_w < W) {
                    for (int ic = 0; ic < IN_C; ++ic) {
                        float w_val = weight[get_index(c, ic, ki, kh, kw, IN_C, K, K, K)];
                        float in_val = 0.0f;
                        // Average over depth dimension
                        for (int id = 0; id < D; ++id) {
                            in_val += input[get_index(b, ic, id, i_h, i_w, IN_C, D, H, W)];
                        }
                        in_val /= D; // Mean over depth
                        sum_val += w_val * in_val;
                    }
                }
            }
        }
    }

    // Add bias
    sum_val += bias[c];

    // We now compute softmax across channel dimension for this spatial location
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int cc = 0; cc < OUT_C; ++cc) {
        float val = 0.0f;
        for (int ki = 0; ki < K; ++ki) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int i_h = h * stride - padding + kh;
                    int i_w = w * stride - padding + kw;
                    if (i_h >= 0 && i_h < H && i_w >= 0 && i_w < W) {
                        for (int ic = 0; ic < IN_C; ++ic) {
                            float w_val = weight[get_index(cc, ic, ki, kh, kw, IN_C, K, K, K)];
                            float in_val = 0.0f;
                            for (int id = 0; id < D; ++id) {
                                in_val += input[get_index(b, ic, id, i_h, i_w, IN_C, D, H, W)];
                            }
                            in_val /= D;
                            val += w_val * in_val;
                        }
                    }
                }
            }
        }
        val += bias[cc];
        max_val = fmaxf(max_val, val);
    }

    // Compute exp(val - max) for numerical stability
    float exp_val = __expf(sum_val - max_val);

    // Sum up all exps
    float exp_sum = 0.0f;
    #pragma unroll
    for (int cc = 0; cc < OUT_C; ++cc) {
        float val = 0.0f;
        for (int ki = 0; ki < K; ++ki) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int i_h = h * stride - padding + kh;
                    int i_w = w * stride - padding + kw;
                    if (i_h >= 0 && i_h < H && i_w >= 0 && i_w < W) {
                        for (int ic = 0; ic < IN_C; ++ic) {
                            float w_val = weight[get_index(cc, ic, ki, kh, kw, IN_C, K, K, K)];
                            float in_val = 0.0f;
                            for (int id = 0; id < D; ++id) {
                                in_val += input[get_index(b, ic, id, i_h, i_w, IN_C, D, H, W)];
                            }
                            in_val /= D;
                            val += w_val * in_val;
                        }
                    }
                }
            }
        }
        val += bias[cc];
        exp_sum += __expf(val - max_val);
    }

    // Softmax: exp_val / exp_sum
    float softmax_out = exp_val / (exp_sum + 1e-10f);

    // Tanh
    float tanh_out = tanhf(softmax_out);

    // Scale
    output[idx] = tanh_out * scaling_factor;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    int stride,
    int padding,
    int K,
    int OUT_D,
    int OUT_H,
    int OUT_W);
"""

# Compile the inline CUDA kernel
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

        # Initialize transposed convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

        # Store output dimensions
        self.OUT_D = 1  # After mean pooling over depth
        self.OUT_H = 128  # H_out = (H_in - 1) * stride - 2 * padding + kernel_size
        self.OUT_W = 128

        # Bind the fused CUDA function
        self.fused_op = fused_op.fused_op_cuda

    def forward(self, x):
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