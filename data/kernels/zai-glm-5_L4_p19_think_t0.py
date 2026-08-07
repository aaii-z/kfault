python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Constants for GELU approximation
constexpr float GELU_COEF = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f; 

// Kernel: Fused ConvTranspose2d + GELU
// Specialized for stride=1, padding=0, dilation=1
__global__ void fused_conv_transpose_gelu_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int N, int C_in, int C_out, int H_in, int W_in, int K) {
    
    int H_out = H_in + K - 1;
    int W_out = W_in + K - 1;

    // Block and thread indices
    int n = blockIdx.x / C_out;
    int oc = blockIdx.x % C_out;
    
    int oh = blockIdx.y * blockDim.x + threadIdx.x;
    int ow = blockIdx.z * blockDim.y + threadIdx.y;

    if (oh >= H_out || ow >= W_out) return;

    float val = bias[oc];

    // Shared memory for input tile
    // Size: (blockDim.x + K - 1) * (blockDim.y + K - 1)
    // blockDim is 16x16, K=3 -> 18x18
    extern __shared__ float shared_input[];

    // Loop over input channels
    for (int ic = 0; ic < C_in; ++ic) {
        // Load input tile into shared memory
        int tile_h_start = blockIdx.y * blockDim.x - K + 1;
        int tile_w_start = blockIdx.z * blockDim.y - K + 1;
        
        int load_idx = threadIdx.y * blockDim.x + threadIdx.x;
        int tile_width = blockDim.y + K - 1;
        int total_tile_size = (blockDim.x + K - 1) * tile_width;
        
        if (load_idx < total_tile_size) {
            int sh_h = load_idx / tile_width;
            int sh_w = load_idx % tile_width;
            
            int ih = tile_h_start + sh_h;
            int iw = tile_w_start + sh_w;
            
            float in_val = 0.0f;
            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                in_val = input[n * C_in * H_in * W_in + ic * H_in * W_in + ih * W_in + iw];
            }
            shared_input[sh_h * tile_width + sh_w] = in_val;
        }
        __syncthreads();

        // Compute convolution
        const float* w_ptr = weight + ic * C_out * K * K + oc * K * K;
        
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                int smem_h = threadIdx.x + K - 1 - kh;
                int smem_w = threadIdx.y + K - 1 - kw;
                
                float w = w_ptr[kh * K + kw];
                float i = shared_input[smem_h * tile_width + smem_w];
                
                val += w * i;
            }
        }
    }

    // Apply GELU (approximation)
    float cdf = 0.5f * (1.0f + tanhf(SQRT_2_OVER_PI * (val + GELU_COEF * val * val * val)));
    val = val * cdf;

    // Write output
    output[n * C_out * H_out * W_out + oc * H_out * W_out + oh * W_out + ow] = val;
}

// Kernel: GroupNorm Stats
__global__ void group_norm_stats_kernel(
    const float* __restrict__ x, 
    float* __restrict__ mean, 
    float* __restrict__ var, 
    int N, int C, int H, int W, int G, int C_per_G) {
    
    int n = blockIdx.x;
    int g = blockIdx.y;
    
    int idx = threadIdx.x;
    int stride = blockDim.x;
    
    int group_start_c = g * C_per_G;
    int total_elements = C_per_G * H * W;
    
    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;
    
    for (int i = idx; i < total_elements; i += stride) {
        int c_local = i / (H * W);
        int hw_local = i % (H * W);
        int h = hw_local / W;
        int w = hw_local % W;
        
        int c_global = group_start_c + c_local;
        
        float val = x[n * C * H * W + c_global * H * W + h * W + w];
        sum_val += val;
        sum_sq_val += val * val;
    }
    
    __shared__ float s_sum[1024];
    __shared__ float s_sum_sq[1024];
    
    s_sum[idx] = sum_val;
    s_sum_sq[idx] = sum_sq_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s) {
            s_sum[idx] += s_sum[idx + s];
            s_sum_sq[idx] += s_sum_sq[idx + s];
        }
        __syncthreads();
    }
    
    if (idx == 0) {
        float mean_val = s_sum[0] / total_elements;
        mean[n * G + g] = mean_val;
        var[n * G + g] = s_sum_sq[0] / total_elements - mean_val * mean_val;
    }
}

// Kernel: GroupNorm Apply
__global__ void group_norm_apply_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ mean, 
    const float* __restrict__ var, 
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    float* __restrict__ y, 
    int N, int C, int H, int W, int G, int C_per_G, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * H * W;
    
    if (idx < total_size) {
        int n = idx / (C * H * W);
        int rem = idx % (C * H * W);
        int c = rem / (H * W);
        int hw = rem % (H * W);
        int h = hw / W;
        int w = hw % W;
        
        int g = c / C_per_G;
        
        float mean_val = mean[n * G + g];
        float var_val = var[n * G + g];
        float inv_std = rsqrtf(var_val + eps);
        
        float val = x[idx];
        float norm_val = (val - mean_val) * inv_std;
        
        y[idx] = norm_val * gamma[c] + beta[c];
    }
}

// Wrappers
torch::Tensor fused_conv_transpose_gelu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int K) {
    
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1);
    
    int H_out = H_in + K - 1;
    int W_out = W_in + K - 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    int block_h = 16;
    int block_w = 16;
    dim3 block(block_h, block_w);
    dim3 grid(N * C_out, (H_out + block_h - 1) / block_h, (W_out