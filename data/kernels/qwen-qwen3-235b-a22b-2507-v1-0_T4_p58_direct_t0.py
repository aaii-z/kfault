import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused ConvTranspose3d + LogSumExp + HardSwish + Subtract + Clamp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Sigmoid approximation using fast tanh-like function
__device__ float fast_sigmoid(float x) {
    return 0.5f + 0.5f * x / (1.0f + fabsf(x));
}

// Custom kernel for fused transposed conv, logsumexp, hardswish, bias subtract, and clamp
// We break it into two main parts: 
// 1. Transposed convolution with logsumexp reduction along channel dim
// 2. Element-wise: x * sigmoid(x+3)/6 - bias, then clamp

extern "C" __global__ void fused_conv_lse_hswish_sub_clamp(
    const float* input,
    const float* weight,
    const float* bias_param,
    float* output,
    // Input tensor shape: [B, C_in, D, H, W]
    int B, int C_in, int D, int H, int W,
    // Output channels and spatial sizes
    int C_out, int D_out, int H_out, int W_out,
    // Conv params
    int kernel_size, int stride, int padding,
    // Transposed conv output padding (assumed 0)
    int output_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * D_out * H_out * W_out;
    
    if (idx >= total_elements) return;

    int b = idx / (D_out * H_out * W_out);
    int d_out = (idx / (H_out * W_out)) % D_out;
    int h_out = (idx / W_out) % H_out;
    int w_out = idx % W_out;

    // Perform transposed convolution manually: iterate over input channels, kernel space
    float conv_val = 0.0f;
    int k = kernel_size;
    int pad = padding;
    int stride_val = stride;

    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kd = 0; kd < k; kd++) {
            for (int kh = 0; kh < k; kh++) {
                for (int kw = 0; kw < k; kw++) {
                    int d_in = d_out * stride_val - pad + kd;
                    int h_in = h_out * stride_val - pad + kh;
                    int w_in = w_out * stride_val - pad + kw;

                    if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        float input_val = input[b * C_in * D * H * W + c_in * D * H * W + d_in * H * W + h_in * W + w_in];
                        float weight_val = weight[c_in * C_out * k * k * k + c_in * k * k * k + kd * k * k + kh * k + kw]; // Assuming grouped conv: weight shape [C_in, C_out, k,k,k] but C_out=1 per group
                        conv_val += input_val * weight_val;
                    }
                }
            }
        }
    }

    // Add bias (per output channel, but we reduce over C_out later)
    // We are going to compute logsumexp over C_out dimension
    // So we store all C_out values in shared memory? -> No, we do reduction in global memory over C_out

    // Instead, we launch one thread per (b, d_out, h_out, w_out), and loop over C_out
    // But we want to fuse logsumexp: so we need to collect all C_out values for this spatial location
    // We use shared memory for block-level reduction

    extern __shared__ float shared_mem[];
    float* shared_vals = shared_mem;
    int tid = threadIdx.x;

    // Each thread handles one C_out
    float local_val = conv_val; // + bias[c_out]; // bias added after conv
    shared_vals[tid] = local_val;
    __syncthreads();

    // Now perform logsumexp across C_out dimension
    // First: find max
    float max_val = -1e30f;
    for (int c = 0; c < C_out; c++) {
        int sh_idx = (tid + c) % blockDim.x;
        if (c < C_out) {
            max_val = fmaxf(max_val, shared_vals[sh_idx]);
        }
    }
    __syncthreads();

    // Sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int c = 0; c < C_out; c++) {
        int sh_idx = (tid + c) % blockDim.x;
        if (c < C_out) {
            sum_exp += expf(shared_vals[sh_idx] - max_val);
        }
    }
    __syncthreads();

    float logsumexp_val = max_val + logf(sum_exp);

    // Now apply HardSwish: x * sigmoid(x + 3) / 6
    float x = logsumexp_val;
    float hswish_val = x * fast_sigmoid(x + 3.0f) / 6.0f;

    // Subtract bias (scalar)
    float biased_val = hswish_val - bias_param[0];

    // Clamp to [-1, 1]
    float result = fmaxf(-1.0f, fminf(1.0f, biased_val));

    // Write to output (only one value per spatial location, since keepdim=True)
    output[b * 1 * D_out * H_out * W_out + 0 * D_out * H_out * W_out + d_out * H_out * W_out + h_out * W_out + w_out] = result;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_param,
    int C_in, int C_out, int D, int H, int W,
    int kernel_size, int stride, int padding
);
"""

# Note: The above kernel is a simplified version. In practice, we would split the kernel into two:
# 1. Transposed conv + logsumexp (with proper grouping and memory layout)
# But due to complexity and inline constraints, we assume a grouped transposed conv where each input channel maps to one output channel group.
# However, PyTorch's ConvTranspose3d is not grouped by default. So we need to handle full matrix.

# Given the complexity of fusing transposed conv with logsumexp in one kernel,
# we instead do a two-kernel approach: custom transposed conv, then fused lse+hswish-sub-clamp

# Let's instead do:
# Kernel 1: Custom transposed conv (reuse PyTorch's weight layout)
# Kernel 2: Fused logsumexp + hardswish + bias subtract + clamp

# But for true fusion and speedup, we do a single kernel that does:
# - Transposed conv for all output channels at a spatial location
# - Then logsumexp reduction over output channels
# - Then hardswish, subtract bias, clamp

# We assume weight layout: (C_in, C_out, k, k, k)
# Input: (B, C_in, D, H, W)
# Output of conv_transpose: (B, C_out, D_out, H_out, W_out)
# Then logsumexp over C_out -> (B, 1, D_out, H_out, W_out)

# We launch one block per (b, d_out, h_out, w_out), and each block computes all C_out values

fused_full_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float fast_sigmoid(float x) {
    return 0.5f + 0.5f * x / (1.0f + fabsf(x));
}

// Fused kernel: transposed conv -> logsumexp(dim=1) -> hardswish -> subtract bias -> clamp
// Grid: (B * D_out * H_out * W_out) blocks
// Block: C_out threads (must be <= 1024)
extern "C" __global__ void fused_transpose_conv_lse_hswish_clamp(
    const float* input,
    const float* weight,
    const float* bias_weight,
    float* output,
    int B, int C_in, int D, int H, int W,
    int C_out, int D_out, int H_out, int W_out,
    int k, int stride, int pad
) {
    int b = blockIdx.x / (D_out * H_out * W_out);
    int d_out = (blockIdx.x / (H_out * W_out)) % D_out;
    int h_out = (blockIdx.x / W_out) % H_out;
    int w_out = blockIdx.x % W_out;

    int c_out = threadIdx.x;
    if (c_out >= C_out) return;

    // Transposed convolution: sum over C_in, kD, kH, kW
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

    // Each thread has one C_out value -> now do block-wide logsumexp
    extern __shared__ float shared_data[];
    float* values = shared_data;
    values[c_out] = conv_val;
    __syncthreads();

    // Block reduction for logsumexp
    float max_val = -1e30f;
    for (int c = 0; c < C_out; c++) {
        max_val = fmaxf(max_val, values[c]);
    }
    __syncthreads();

    float sum_exp = 0.0f;
    for (int c = 0; c < C_out; c++) {
        sum_exp += expf(values[c] - max_val);
    }
    __syncthreads();

    if (c_out == 0) {
        float lse_val = max_val + logf(sum_exp);
        float hswish_val = lse_val * fast_sigmoid(lse_val + 3.0f) / 6.0f;
        float result = fmaxf(-1.0f, fminf(1.0f, hswish_val - bias_weight[0]));
        output[b * D_out * H_out * W_out + d_out * H_out * W_out + h_out * W_out + w_out] = result;
    }
}
"""

fused_cpp_full = """
torch::Tensor fused_transpose_conv_lse_hswish_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_weight,
    int C_in, int C_out, int D, int H, int W,
    int k, int stride, int pad
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_transpose_conv_lse_hswish_clamp",
    cpp_sources=[fused_cpp_full],
    cuda_sources=[fused_full_source],
    functions=["fused_transpose_conv_lse_hswish_clamp_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"],
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
        # Extract parameters
        weight = self.conv_transpose.weight  # (C_out, C_in, k, k, k)
        bias_weight = self.bias
        B, C_in, D, H, W = x.shape
        k = self.kernel_size
        stride = self.stride
        pad = self.padding

        # Compute output shape for transposed conv
        D_out = (D - 1) * stride - 2 * pad + k
        H_out = (H - 1) * stride - 2 * pad + k
        W_out = (W - 1) * stride - 2 * pad + k

        # Launch one block per output spatial location
        total_blocks = B * D_out * H_out * W_out
        block_size = self.out_channels
        shared_mem = self.out_channels * 4  # 4 bytes per float

        # Ensure block size is within limits
        if block_size > 1024:
            block_size = 1024

        # Call fused CUDA kernel
        output = self.fused_op.fused_transpose_conv_lse_hswish_clamp_cuda(
            x, weight, bias_weight,
            C_in, self.out_channels, D, H, W,
            k, stride, pad
        )

        # Reshape output to (B, 1, D_out, H_out, W_out)
        output = output.view(B, 1, D_out, H_out, W_out)
        return output