import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for fused ConvTranspose3d + ReLU + GroupNorm
fused_op_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose_relu_gn_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int num_groups, float eps) {
    
    // Dynamic shared memory allocation
    // Layout: [Input Patch (C_in * k_vol)] [Intermediate (C_out)] [Stats (2 * num_groups)]
    extern __shared__ float s_mem[];
    
    int patch_size = C_in * kD * kH * kW;
    float* s_input_patch = s_mem;
    float* s_intermediate = s_mem + patch_size;
    float* s_stats = s_intermediate + C_out;

    // Grid mapping: blockIdx.x encodes (N, D_out, H_out, W_out)
    int spatial_idx = blockIdx.x;
    int w_out = spatial_idx % W_out;
    spatial_idx /= W_out;
    int h_out = spatial_idx % H_out;
    spatial_idx /= H_out;
    int d_out = spatial_idx % D_out;
    int n = spatial_idx / D_out;

    int c_out = threadIdx.x; // 0 to C_out - 1
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // 1. Cooperative Load of Input Patch
    // Each thread loads multiple elements to cover the whole patch
    for (int i = tid; i < patch_size; i += bdim) {
        // Decode linear index i to (c_in, kd, kh, kw)
        int kw = i % kW;
        int rem = i / kW;
        int kh = rem % kH;
        rem /= kH;
        int kd = rem % kD;
        int c_in = rem / kD;

        // Calculate input coordinates
        int d_in = d_out - kd;
        int h_in = h_out - kh;
        int w_in = w_out - kw;

        float val = 0.0f;
        if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            long long in_idx = (long long)n * C_in * D_in * H_in * W_in + 
                               c_in * D_in * H_in * W_in + 
                               d_in * H_in * W_in + 
                               h_in * W_in + 
                               w_in;
            val = __ldg(&input[in_idx]);
        }
        s_input_patch[i] = val;
    }
    __syncthreads();

    // 2. ConvTranspose + ReLU
    // Compute output for channel c_out
    float sum = 0.0f;
    int kernel_vol = kD * kH * kW;

    // Weight layout: (C_in, C_out, kD, kH, kW)
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int patch_base = c_in * kernel_vol;
        long long weight_base = (long long)c_in * C_out * kernel_vol + c_out * kernel_vol;
        
        for (int k = 0; k < kernel_vol; ++k) {
            sum += s_input_patch[patch_base + k] * __ldg(&weight[weight_base + k]);
        }
    }
    
    // ReLU
    sum = fmaxf(sum, 0.0f);

    // 3. GroupNorm
    // Store intermediate result to shared memory
    if (c_out < C_out) {
        s_intermediate[c_out] = sum;
    }
    __syncthreads();

    // Compute stats for each group
    int channels_per_group = C_out / num_groups;
    int group_id = c_out / channels_per_group;
    int lane_id = c_out % channels_per_group;

    // Use thread 0 of each group (lane_id == 0) to compute mean and var
    if (lane_id == 0) {
        float g_sum = 0.0f;
        float g_sum_sq = 0.0f;
        for (int i = 0; i < channels_per_group; ++i) {
            float val = s_intermediate[group_id * channels_per_group + i];
            g_sum += val;
            g_sum_sq += val * val;
        }
        float mean = g_sum / channels_per_group;
        float var = g_sum_sq / channels_per_group - mean * mean;
        
        s_stats[group_id * 2] = mean;
        s_stats[group_id * 2 + 1] = var;
    }
    __syncthreads();

    // Apply normalization
    float mean = s_stats[group_id * 2];
    float var = s_stats[group_id * 2 + 1];
    float norm_val = (sum - mean) * rsqrtf(var + eps);
    
    // Scale and shift
    float final_val = norm_val * gamma[c_out] + beta[c_out];

    // Write output
    if (c_out < C_out) {
        long long out_idx = (long long)n * C_out * D_out * H_out * W_out + 
                            c_out * D_out * H_out * W_out + 
                            d_out * H_out * W_out + 
                            h_out * W_out + 
                            w_out;
        output[out_idx] = final_val;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups, float eps) {
    
    // Dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int C_out = weight.size(1);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int D_out = D_in + kD - 1;
    int H_out = H_in + kH - 1;
    int W_out = W_in + kW - 1;

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    int spatial_size = N * D_out * H_out * W_out;
    int threads = C_out; // Assuming C_out <= 1024
    
    // Shared memory size calculation
    int patch_size = C_in * kD * kH * kW;
    int shared_mem_size = (patch_size + C_out + 2 * num_groups) * sizeof(float);

    // Launch kernel
    fused_conv_transpose_relu_gn_kernel<<<spatial_size, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        kD, kH, kW,
        num_groups, eps
    );

    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        
        # Define parameters manually to replace nn.ConvTranspose3d and nn.GroupNorm
        # ConvTranspose3d weights
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        
        # GroupNorm weights (gamma and beta)
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights similar to PyTorch defaults
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.groups = groups
        self.eps = 1e-5
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(x, self.weight, self.gamma, self.beta, self.groups, self.eps)