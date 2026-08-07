import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Fused Bias + Scale + Sigmoid
__global__ void fused_bias_scale_sigmoid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;

    if (idx < total_elements) {
        // Layout is NCHW
        // Calculate channel index
        int c = (idx / (H * W)) % C;
        
        float val = input[idx];
        float b = bias[c];
        float s = scale[c];
        
        // (x + bias) * scale
        val = (val + b) * s;
        
        // Sigmoid: 1 / (1 + exp(-x))
        val = 1.0f / (1.0f + expf(-val));
        
        output[idx] = val;
    }
}

// Kernel 2: GroupNorm
template <int BLOCK_SIZE>
__global__ void group_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C, int H, int W, int G, int C_per_G) {
    
    int n = blockIdx.x;
    int g = blockIdx.y;
    
    int group_start_c = g * C_per_G;
    
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_sum_sq[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    int group_size = C_per_G * H * W;
    int base_offset = n * C * H * W;
    
    // First pass: compute mean and variance
    for (int i = tid; i < group_size; i += BLOCK_SIZE) {
        int c_local = i / (H * W);
        int hw_rem = i % (H * W);
        int h = hw_rem / W;
        int w = hw_rem % W;
        
        int c_global = group_start_c + c_local;
        int idx = base_offset + c_global * H * W + h * W + w;
        
        float val = input[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    float mean = s_sum[0] / group_size;
    float var = s_sum_sq[0] / group_size - mean * mean;
    float inv_std = rsqrtf(var + 1e-5f);
    
    // Second pass: normalize
    for (int i = tid; i < group_size; i += BLOCK_SIZE) {
        int c_local = i / (H * W);
        int hw_rem = i % (H * W);
        int h = hw_rem / W;
        int w = hw_rem % W;
        
        int c_global = group_start_c + c_local;
        int idx = base_offset + c_global * H * W + h * W + w;
        
        float val = input[idx];
        float normalized = (val - mean) * inv_std;
        output[idx] = normalized * gamma[c_global] + beta[c_global];
    }
}

// C++ Interface
torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor input, torch::Tensor bias, torch::Tensor scale) {
    
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::empty_like(input);
    
    int total_elements = input.numel();
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    return output;
}

torch::Tensor group_norm_cuda(
    torch::Tensor input, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    int G) {
    
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    int C_per_G = C / G;
    
    auto output = torch::empty_like(input);
    
    dim3 grid(N, G);
    int block_size = 256;
    
    group_norm_kernel<256><<<grid, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, G, C_per_G
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);
torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int G);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_bias_scale_sigmoid_cuda", "group_norm_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.custom_ops = custom_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.custom_ops.fused_bias_scale_sigmoid_cuda(x, self.bias, self.scale)
        x = self.custom_ops.group_norm_cuda(x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups)
        return x