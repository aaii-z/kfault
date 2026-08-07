import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused ConvTranspose2d + min(dim=1) + sum(dim=2) + GELU + bias addition
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation approximation: x * 0.5 * (1 + erf(x * 0.7071067811865476))
__device__ float gelu(float x) {
    return x * 0.5f * (1.0f + erf(x * 0.7071067811865476f));
}

__global__ void fused_conv_transpose_min_sum_gelu_bias(
    const float* input,
    const float* weight,
    const float* bias_vec,
    float* output,
    // Input tensor shape: [B, C_in, H_in, W_in]
    int B, int C_in, int H_in, int W_in,
    // Output of conv_transpose: [B, C_out, H_out, W_out]
    int C_out, int H_out, int W_out,
    // Conv params
    int kernel_size, int stride, int padding, int output_padding,
    // Weight shape: [C_in, C_out, k, k]
    int weight_stride_0, int weight_stride_1, int weight_stride_2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * H_out * W_out;
    
    if (idx >= total_elements) return;

    int b = idx / (H_out * W_out);
    int h_out = (idx % (H_out * W_out)) / W_out;
    int w_out = idx % W_out;

    // Compute conv_transpose: iterate over input channels, output channels, and kernel
    float* temp_buffer = new float[C_out];
    for (int co = 0; co < C_out; co++) {
        float val = 0.0f;
        for (int ci = 0; ci < C_in; ci++) {
            for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                    int h_in = h_out - padding + ki * stride;
                    int w_in = w_out - padding + kj * stride;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int input_idx = b * C_in * H_in * W_in + ci * H_in * W_in + h_in * W_in + w_in;
                        int weight_idx = ci * weight_stride_0 + co * weight_stride_1 + ki * weight_stride_2 + kj;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        temp_buffer[co] = val;
    }

    // Min over channel dimension (dim=1): find min across C_out
    float min_val = temp_buffer[0];
    for (int co = 1; co < C_out; co++) {
        if (temp_buffer[co] < min_val) {
            min_val = temp_buffer[co];
        }
    }

    // Sum over height dimension (dim=2): but we are at a specific h_out, so we will later reduce over H_out
    // However, note: the sum(dim=2) collapses H dimension -> output has H=1
    // So we are writing to a tensor of size [B, 1, 1, W_out]
    // But we are fusing the entire computation: we need to accumulate sum over all h_out later?
    // Actually, we must restructure: we cannot do sum over H_out in this kernel if we are parallelizing over H_out.
    // Instead, we do: after conv_transpose -> min over C -> sum over H -> GELU -> add bias
    // So we need to first compute all (b, h_out, w_out), then reduce over h_out.

    // We change strategy: this kernel will not do the sum. We split into two kernels.
    // Let's instead do: Kernel 1: conv_transpose + min(C) -> [B, 1, H_out, W_out]
    // Kernel 2: sum(H) + GELU + bias -> [B, 1, 1, W_out]

    // But we want full fusion. So we need to restructure the algorithm.

    // Alternative: we do not fuse conv_transpose with the reductions. Instead, we fuse the post-processing.

    // Given complexity, we instead create a two-stage kernel:
    // Stage 1: ConvTranspose -> Min over channel -> output to temp tensor [B, 1, H_out, W_out]
    // Stage 2: Sum over H, then GELU, then bias add -> [B, 1, 1, W_out]

    // But we are allowed to write multiple kernels. Let's do full fusion in two steps.

    // Actually, we decide to replace only the post-processing: min + sum + gelu + add with a fused kernel,
    // and leave conv_transpose as is, because PyTorch's ConvTranspose2d is already highly optimized.

    // So we will not include conv_transpose in the custom kernel.

    // This code will be replaced by a simpler fused kernel for the sequence:
    // min(dim=1) -> sum(dim=2) -> gelu -> add(bias)

    // So we comment out the above and write a new kernel below for the fusion of min+sum+gelu+bias.

    delete[] temp_buffer;
}
"""

# Instead, we write a clean fused kernel for: min(dim=1) + sum(dim=2) + gelu + add(bias)
# Input: [B, C, H, W] -> min(dim=1, keepdim=True) -> [B, 1, H, W]
#        -> sum(dim=2, keepdim=True) -> [B, 1, 1, W]
#        -> gelu -> [B, 1, 1, W]
#        -> + bias [1,1,1] -> [B,1,1,W]

fused_postprocess_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// GELU activation approximation
__device__ float gelu(float x) {
    return x * 0.5f * (1.0f + erf(x * 0.7071067811865476f));
}

extern "C" __global__ void fused_min_sum_gelu_bias(
    const float* input,
    const float* bias,
    float* output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * W;
    if (idx >= total_elements) return;

    int b = idx / W;
    int w = idx % W;

    // Step 1: min over channel dimension (dim=1) -> [B, H, W]
    float min_val = -FLT_MAX;
    for (int c = 0; c < C; c++) {
        float val = input[b * C * H * W + c * H * W + 0 * W + w]; // first h
        if (c == 0 || val < min_val) {
            min_val = val;
        }
        // We need to check all H
        for (int h = 1; h < H; h++) {
            float val_h = input[b * C * H * W + c * H * W + h * W + w];
            if (val_h < min_val) {
                min_val = val_h;
            }
        }
    }

    // But wait: min is per (b, h, w) -> [B, H, W], then sum over H -> [B, W]
    // So we must first compute min over C for each (b, h, w), then sum over H.

    float sum_val = 0.0f;
    for (int h = 0; h < H; h++) {
        float min_at_hw = input[b * C * H * W + 0 * H * W + h * W + w]; // c=0
        for (int c = 1; c < C; c++) {
            float val = input[b * C * H * W + c * H * W + h * W + w];
            if (val < min_at_hw) {
                min_at_hw = val;
            }
        }
        sum_val += min_at_hw;
    }

    // Apply GELU
    float gelu_val = gelu(sum_val);

    // Add bias (scalar)
    gelu_val += bias[0];

    // Output: [B, 1, 1, W] -> index: b * W + w
    output[b * W + w] = gelu_val;
}
"""

fused_postprocess_cpp = """
void fused_min_sum_gelu_bias(const float* input, const float* bias, float* output, int B, int C, int H, int W);
"""

# Compile the fused post-processing kernel
fused_ops = load_inline(
    name="fused_postprocess",
    cpp_sources=[fused_postprocess_cpp],
    cuda_sources=[fused_postprocess_source],
    functions=["fused_min_sum_gelu_bias"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for min + sum + gelu + bias.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_kernel = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)  # [B, C_out, H_out, W_out]

        B, C, H, W = x.shape

        # Flatten input for kernel
        input_flat = x.contiguous()

        # Output: [B, 1, 1, W]
        output = torch.empty((B, 1, 1, W), dtype=torch.float32, device=x.device)

        # Flatten output
        output_flat = output.view(-1)  # [B*W]

        # Launch kernel
        block_size = 256
        num_blocks = (B * W + block_size - 1) // block_size

        self.fused_kernel.fused_min_sum_gelu_bias(
            input_flat.data_ptr(),
            self.bias.data_ptr(),
            output_flat.data_ptr(),
            B, C, H, W
        )

        return output