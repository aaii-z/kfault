import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for BatchNorm + Mean Subtraction fusion
fused_bn_mean_sub_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel for Batch Normalization Forward Pass
// Computes mean and variance, and normalizes the input
__global__ void batch_norm_forward_kernel(
    const float* __restrict__ x, 
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    int N, int C, int D, int H, int W, 
    float eps) {
    
    int c = blockIdx.x;
    int spatial_size = D * H * W;
    int n = blockIdx.y;
    
    if (c >= C) return;

    float mean = running_mean[c];
    float var = running_var[c];
    float scale = weight[c] / sqrtf(var + eps);
    float shift = bias[c] - mean * scale;

    int base_idx = (n * C + c) * spatial_size;
    
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int idx = base_idx + i;
        out[idx] = x[idx] * scale + shift;
    }
}

// Kernel to compute the mean of the spatial dimensions (D, H, W)
// and subtract it from the input tensor.
__global__ void mean_subtract_kernel(
    float* __restrict__ x,
    int N, int C, int D, int H, int W) {

    int n = blockIdx.x;
    int c = blockIdx.y;
    
    if (n >= N || c >= C) return;

    int spatial_size = D * H * W;
    int base_idx = (n * C + c) * spatial_size;

    // 1. Compute sum of spatial elements
    float sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        sum += x[base_idx + i];
    }

    // Block reduction for sum
    __shared__ float s_sum[256]; // Assuming block size <= 256
    s_sum[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // 2. Compute mean
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        mean = s_sum[0] / spatial_size;
    }
    // Broadcast mean to all threads in the block
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = mean;
    }
    __syncthreads();
    
    // 3. Subtract mean
    float val_mean = s_mean;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        x[base_idx + i] -= val_mean;
    }
}

// Helper function to launch the fused operation
torch::Tensor fused_bn_mean_sub_cuda(
    torch::Tensor x, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    float eps) {

    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    auto out = torch::empty_like(x);

    // Launch BatchNorm kernel
    // Grid: (C, N), Block: 256
    dim3 blocks_bn(C, N);
    int threads_bn = 256;
    
    batch_norm_forward_kernel<<<blocks_bn, threads_bn>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        N, C, D, H, W, eps
    );

    // Launch Mean Subtraction kernel
    // Grid: (N, C), Block: 256
    dim3 blocks_mean(N, C);
    int threads_mean = 256;
    
    mean_subtract_kernel<<<blocks_mean, threads_mean>>>(
        out.data_ptr<float>(),
        N, C, D, H, W
    );

    return out;
}
"""

fused_bn_mean_sub_cpp_source = (
    "torch::Tensor fused_bn_mean_sub_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, float eps);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_bn_mean_sub",
    cpp_sources=fused_bn_mean_sub_cpp_source,
    cuda_sources=fused_bn_mean_sub_source,
    functions=["fused_bn_mean_sub_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model with custom CUDA kernel for fused BatchNorm and Mean Subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # We keep the BatchNorm3d module to hold the parameters and running stats
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Extract BatchNorm parameters
        bn = self.batch_norm
        weight = bn.weight
        bias = bn.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps
        
        # Call the custom fused CUDA kernel
        x = self.fused_op.fused_bn_mean_sub_cuda(x, weight, bias, running_mean, running_var, eps)
        
        return x