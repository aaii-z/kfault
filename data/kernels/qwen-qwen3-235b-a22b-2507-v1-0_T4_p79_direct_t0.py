import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D convolution, element-wise multiply, instance norm, clamp, multiply, and max
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to calculate 3D index
__device__ int get_index(int d, int h, int w, int D, int H, int W) {
    return d * H * W + h * W + w;
}

__global__ void fused_conv_mul_norm_clamp_max_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* multiplier,
    float* output,
    int B, int C_in, int C_out, int D, int H, int W,
    int kD, int kH, int kW,
    int pad_d, int pad_h, int pad_w,
    float clamp_min, float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * D * H * W;
    
    if (idx >= total_elements) return;

    int b = idx / (D * H * W);
    int d = (idx / (H * W)) % D;
    int h = (idx / W) % H;
    int w = idx % W;

    int C_out_per_thread = (C_out + gridDim.y - 1) / gridDim.y;
    int c_start = blockIdx.y * C_out_per_thread;
    int c_end = min(c_start + C_out_per_thread, C_out);

    float sum_val;
    float mean = 0.0f, var = 0.0f;

    // First pass: compute conv and accumulate for mean
    for (int c_out = c_start; c_out < c_end; ++c_out) {
        sum_val = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kd = 0; kd < kD; ++kd) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int in_d = d + kd - pad_d;
                        int in_h = h + kh - pad_h;
                        int in_w = w + kw - pad_w;
                        if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                            int in_idx = b * C_in * D * H * W + c_in * D * H * W + in_d * H * W + in_h * W + in_w;
                            int weight_idx = c_out * C_in * kD * kH * kW + c_in * kD * kH * kW + kd * kH * kW + kh * kW + kw;
                            sum_val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        sum_val += bias[c_out];

        // Multiply by multiplier
        float x = sum_val * multiplier[c_out];

        // Accumulate for mean
        mean += x;
    }

    mean /= (c_end - c_start);

    // Second pass: compute variance
    for (int c_out = c_start; c_out < c_end; ++c_out) {
        sum_val = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kd = 0; kd < kD; ++kd) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int in_d = d + kd - pad_d;
                        int in_h = h + kh - pad_h;
                        int in_w = w + kw - pad_w;
                        if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                            int in_idx = b * C_in * D * H * W + c_in * D * H * W + in_d * H * W + in_h * W + in_w;
                            int weight_idx = c_out * C_in * kD * kH * kW + c_in * kD * kH * kW + kd * kH * kW + kh * kW + kw;
                            sum_val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        sum_val += bias[c_out];
        float x = sum_val * multiplier[c_out];
        var += (x - mean) * (x - mean);
    }

    var /= (c_end - c_start);
    float eps = 1e-5f;
    float inv_std = rsqrtf(var + eps);

    // Third pass: apply instance norm, clamp, multiply again, and write to temp buffer
    for (int c_out = c_start; c_out < c_end; ++c_out) {
        sum_val = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kd = 0; kd < kD; ++kd) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int in_d = d + kd - pad_d;
                        int in_h = h + kh - pad_h;
                        int in_w = w + kw - pad_w;
                        if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                            int in_idx = b * C_in * D * H * W + c_in * D * H * W + in_d * H * W + in_h * W + in_w;
                            int weight_idx = c_out * C_in * kD * kH * kW + c_in * kD * kH * kW + kd * kH * kW + kh * kW + kw;
                            sum_val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        sum_val += bias[c_out];
        float x = sum_val * multiplier[c_out];
        x = (x - mean) * inv_std;
        x = fmaxf(clamp_min, fminf(clamp_max, x));
        x = x * multiplier[c_out];

        // Store in temporary location for max reduction later
        int temp_idx = b * (D * H * W) * gridDim.y + blockIdx.y * (D * H * W) + d * H * W + h * W + w;
        if (c_out == c_start) {
            output[temp_idx] = x;
        } else {
            output[temp_idx] = fmaxf(output[temp_idx], x);
        }
    }
}

// Kernel to reduce across the C dimension blocks
__global__ void reduce_max_across_blocks(float* temp_output, float* final_output, int total_spatial, int num_blocks_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_spatial) return;

    float max_val = -FLT_MAX;
    for (int bid = 0; bid < num_blocks_y; ++bid) {
        float val = temp_output[bid * total_spatial + idx];
        max_val = fmaxf(max_val, val);
    }
    final_output[idx] = max_val;
}

torch::Tensor fused_conv_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    float clamp_min,
    float clamp_max
) {
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    const int C_out = weight.size(0);
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int pad_d = kD / 2;
    const int pad_h = kH / 2;
    const int pad_w = kW / 2;

    // Output spatial dimensions (same padding assumed)
    const int out_D = D;
    const int out_H = H;
    const int out_W = W;
    const int total_spatial = B * out_D * out_H * out_W;

    // We'll split C_out across Y-dimension of grid
    const int max_blocks_y = 16;
    const int blocks_y = min(max_blocks_y, (C_out + 15) / 16);  // roughly 16 channels per block
    const int threads_per_block = 256;
    const int blocks_x = (total_spatial + threads_per_block - 1) / threads_per_block;

    // Temporary buffer to hold intermediate max values per block
    auto options = input.options();
    auto temp_output = torch::zeros({blocks_y, B, out_D, out_H, out_W}, options);

    // Launch fused kernel
    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_per_block);

    fused_conv_mul_norm_clamp_max_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        temp_output.data_ptr<float>(),
        B, C_in, C_out, D, H, W,
        kD, kH, kW,
        pad_d, pad_h, pad_w,
        clamp_min, clamp_max
    );

    // Now reduce across the block dimension
    auto final_output = torch::zeros({B, out_D, out_H, out_W}, options);
    reduce_max_across_blocks<<<(total_spatial + 255) / 256, 256>>>(
        temp_output.data_ptr<float>(),
        final_output.data_ptr<float>(),
        total_spatial,
        blocks_y
    );

    return final_output;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    float clamp_min,
    float clamp_max
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_op"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model using fused CUDA kernel for:
    Conv3d -> mul -> InstanceNorm3d -> clamp -> mul -> max(dim=1)
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Initialize weights and bias like Conv3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

        # Bind the fused CUDA op
        self.fused_op = fused_ops.fused_conv_op

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias, self.multiplier, self.clamp_min, self.clamp_max)