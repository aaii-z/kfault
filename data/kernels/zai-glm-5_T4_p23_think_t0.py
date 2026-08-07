import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for Conv3d, GroupNorm, and Global Mean fusion
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Conv3d
// Simple direct convolution kernel. Each thread computes one output element.
// Assuming stride=1, padding=0, dilation=1.
__global__ void conv3d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int N, int C_in, int C_out, 
    int D_in, int H_in, int W_in, 
    int kD, int kH, int kW) {
    
    int D_out = D_in - kD + 1;
    int H_out = H_in - kH + 1;
    int W_out = W_in - kW + 1;
    
    int total_out = N * C_out * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_out) {
        // Decode linear index to (n, c_out, d_out, h_out, w_out)
        int w_out = idx % W_out;
        int tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int d_out = tmp % D_out;
        tmp /= D_out;
        int c_out = tmp % C_out;
        int n = tmp / C_out;
        
        float val = bias[c_out];
        
        // Convolution loop
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kd = 0; kd < kD; ++kd) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int d_in = d_out + kd;
                        int h_in = h_out + kh;
                        int w_in = w_out + kw;
                        
                        // Input layout: (N, C_in, D, H, W)
                        int input_idx = ((n * C_in + c_in) * D_in + d_in) * H_in * W_in + h_in * W_in + w_in;
                        // Weight layout: (C_out, C_in, kD, kH, kW)
                        int weight_idx = ((c_out * C_in + c_in) * kD + kd) * kH * kW + kh * kW + kw;
                        
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[idx] = val;
    }
}

// Kernel 2: GroupNorm Stats (Sum and SumSq)
// Grid: (N, G). Block: 1D.
// Each block computes stats for one group instance.
__global__ void gn_stats_kernel(
    const float* __restrict__ input, 
    float* __restrict__ mean_out, 
    float* __restrict__ var_out,
    int N, int C, int D, int H, int W, int G, int C_per_G) {
    
    int n = blockIdx.y; // Batch index
    int g = blockIdx.x; // Group index
    
    int c_start = g * C_per_G;
    int c_end = c_start + C_per_G;
    
    int total_elements = C_per_G * D * H * W;
    
    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;
    
    // Stride loop over the data
    // We map linear thread index to (c_local, d, h, w)
    int linear_idx = threadIdx.x;
    int stride = blockDim.x;
    
    // Calculate total size for this group instance
    int group_size = C_per_G * D * H * W;
    
    while (linear_idx < group_size) {
        int w_idx = linear_idx % W;
        int tmp = linear_idx / W;
        int h_idx = tmp % H;
        tmp /= H;
        int d_idx = tmp % D;
        int c_local = tmp / D;
        
        int c_global = c_start + c_local;
        
        // Global input index: (n, c_global, d_idx, h_idx, w_idx)
        int input_idx = ((n * C + c_global) * D + d_idx) * H * W + h_idx * W + w_idx;
        
        float val = input[input_idx];
        sum_val += val;
        sum_sq_val += val * val;
        
        linear_idx += stride;
    }
    
    // Shared memory reduction
    __shared__ float s_sum[256]; 
    __shared__ float s_sum_sq[256];
    
    s_sum[threadIdx.x] = sum_val;
    s_sum_sq[threadIdx.x] = sum_sq_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float mean = s_sum[0] / total_elements;
        float var = s_sum_sq[0] / total_elements - mean * mean;
        
        int out_idx = n * G + g;
        mean_out[out_idx] = mean;
        var_out[out_idx] = var;
    }
}

// Kernel 3: Apply GroupNorm and Global Mean
// Grid: (N). Block: 1D.
// Each block computes the sum of the normalized values for one batch item.
__global__ void gn_apply_mean_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ mean_in, 
    const float* __restrict__ var_in,
    const float* __restrict__ gamma, 
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C, int D, int H, int W, int G, int C_per_G, float eps) {
    
    int n = blockIdx.x; // Batch index
    
    float total_sum = 0.0f;
    int total_elements = C * D * H * W;
    
    int linear_idx = threadIdx.x;
    int stride = blockDim.x;
    
    while (linear_idx < total_elements) {
        // Decode (c, d, h, w)
        int w_idx = linear_idx % W;
        int tmp = linear_idx / W;
        int h_idx = tmp % H;
        tmp /= H;
        int d_idx = tmp % D;
        int c = tmp / D;
        
        int g = c / C_per_G;
        
        // Load stats
        float mean = mean_in[n * G + g];
        float var = var_in[n * G + g];
        
        // Load input
        int input_idx = ((n * C + c) * D + d_idx) * H * W + h_idx * W + w_idx;
        float val = input[input_idx];
        
        // Normalize
        float val_norm = (val - mean) / sqrtf(var + eps);
        
        // Affine
        float val_out = val_norm * gamma[c] + beta[c];
        
        total_sum += val_out;
        
        linear_idx += stride;
    }
    
    // Reduction
    __shared__ float s_sum[256];
    s_sum[threadIdx.x] = total_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[n] = s_sum[0] / total_elements;
    }
}

// Wrapper functions
torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Ensure contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int C_out = weight.size(0);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);
    
    int D_out = D_in - kD + 1;
    int H_out = H_in - kH + 1;
    int W_out = W_in - kW + 1;
    
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());
    
    int total_out = N * C_out * D_out * H_out * W_out;
    int block_size = 256;
    int num_blocks = (total_out + block_size - 1) / block_size;
    
    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, C_out, D_in, H_in, W_in, kD, kH, kW);
        
    return output;
}

torch::Tensor fused_gn_mean_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps) {
    
    x = x.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();

    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int G = num_groups;
    int C_per_G = C / G;
    
    // Intermediate buffers for stats
    auto mean_buf = torch::zeros({N, G}, x.options());
    auto var_buf = torch::zeros({N, G}, x.options());
    
    // Output
    auto output = torch::zeros({N}, x.options());
    
    // Launch stats kernel
    // Grid: (G, N) -> blockIdx.x = G, blockIdx.y = N
    dim3 blocks(G, N);
    int block_size_stats = 256; 
    
    gn_stats_kernel<<<blocks, block_size_stats>>>(
        x.data_ptr<float>(), mean_buf.data_ptr<float>(), var_buf.data_ptr<float>(),
        N, C, D, H, W, G, C_per_G);
        
    // Launch apply + mean kernel
    // Grid: N
    int block_size_apply = 256;
    
    gn_apply_mean_kernel<<<N, block_size_apply>>>(
        x.data_ptr<float>(), mean_buf.data_ptr<float>(), var_buf.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W, G, C_per_G, eps);
        
    return output;
}
"""

cpp_source = """
torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor fused_gn_mean_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv3d_cuda", "fused_gn_mean_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Define weights manually to match PyTorch initialization
        # Conv3d weights
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(out_channels))
        
        # GroupNorm weights (gamma and beta)
        self.gn_weight = nn.Parameter(torch.ones(out_channels)) # gamma
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))   # beta
        
        # Initialize weights similar to PyTorch defaults
        # PyTorch Conv3d uses kaiming uniform
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        if self.conv_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.conv_bias, -bound, bound)

    def forward(self, x):
        # Run custom Conv3d
        x = custom_ops.conv3d_cuda(x, self.conv_weight, self.conv_bias)
        
        # Run custom Fused GroupNorm + Mean
        # GroupNorm default eps is 1e-5
        x = custom_ops.fused_gn_mean_cuda(x, self.gn_weight, self.gn_bias, self.num_groups, 1e-5)
        
        return x

import math