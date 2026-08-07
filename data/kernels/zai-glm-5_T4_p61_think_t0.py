import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose3d + ReLU + GroupNorm
fused_op_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 128

__global__ void fused_conv_transpose_relu_groupnorm_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int group_size, float eps) {
    
    // Shared memory for input patch
    extern __shared__ float input_patch[];
    
    // Calculate global output coordinates
    // Grid is flattened: (N, D_out, H_out, W_out)
    int idx = blockIdx.x;
    int w_out = idx % W_out;
    idx /= W_out;
    int h_out = idx % H_out;
    idx /= H_out;
    int d_out = idx % D_out;
    int n = idx / D_out;
    
    int c_out = threadIdx.x; // 0 to C_out-1
    
    // Cooperative load of input patch
    int kernel_volume = kD * kH * kW;
    int patch_size = C_in * kernel_volume;
    
    for (int i = threadIdx.x; i < patch_size; i += BLOCK_SIZE) {
        int kw = i % kW;
        int rem = i / kW;
        int kh = rem % kH;
        rem /= kH;
        int kd = rem % kD;
        int c_in = rem / kD;
        
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
        input_patch[i] = val;
    }
    __syncthreads();
    
    // ConvTranspose + ReLU
    float sum = 0.0f;
    
    // Weight layout: (C_in, C_out, kD, kH, kW)
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int patch_base = c_in * kernel_volume;
        long long weight_base = (long long)c_in * C_out * kernel_volume + c_out * kernel_volume;
        
        for (int k = 0; k < kernel_volume; ++k) {
            float in_val = input_patch[patch_base + k];
            float w_val = __ldg(&weight[weight_base + k]);
            sum += in_val * w_val;
        }
    }
    
    // ReLU
    sum = fmaxf(sum, 0.0f);
    
    // GroupNorm
    int group