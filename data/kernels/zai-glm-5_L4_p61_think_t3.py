import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ReLU + GroupNorm
fused_gn_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void group_norm_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N, int C, int D, int H, int W, int G, scalar_t eps) {
    
    int batch_idx = blockIdx.x / G;
    int group_idx = blockIdx.x % G;
    
    int C_per_G = C / G;
    int channel_start = group_idx * C_per_G;
    
    int spatial_size = D * H * W;
    int group_size = C_per_G * spatial_size;
    
    const scalar_t* input_ptr = input + batch_idx * C * spatial_size + channel_start * spatial_size;
    scalar_t* output_ptr = output + batch_idx * C * spatial_size + channel_start * spatial_size;
    
    extern __shared__ char shared_mem[];
    scalar_t* s_sum = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* s_sum_sq = s_sum + blockDim.x;
    
    int tid = threadIdx.x;
    scalar_t local_sum = 0;
    scalar_t local_sum_sq = 0;
    
    // Pass 1: Compute Sum and Sum of Squares (with ReLU)
    for (int i = tid; i < group_size; i += blockDim.x) {
        scalar_t val = input_ptr[i];
        scalar_t val_relu = val > 0 ? val : 0;
        local_sum += val_relu;
        local_sum_sq += val_relu * val_relu;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    scalar_t mean = 0;
    scalar_t var = 0;
    
    if (tid == 0) {
        scalar_t sum = s_sum[0];
        scalar_t sum_sq = s_sum_sq[0];
        mean = sum / group_size;
        var = sum_sq / group_size - mean * mean;
        s_sum[0] = mean;
        s_sum[1] = var;
    }
    __syncthreads();
    
    mean = s_sum[0];
    var = s_sum[1];
    scalar_t inv_std = 1.0 / sqrt(var + eps);
    
    // Pass 2: Normalize
    for (int i = tid; i < group_size; i += blockDim.x) {
        int c_local = i / spatial_size;
        int c_global = channel_start + c_local;
        
        scalar_t val = input_ptr[i];
        scalar_t val_relu = val > 0 ? val : 0;
        
        scalar_t w = weight[c_global];
        scalar_t b = bias[c_global];
        
        output_ptr[i] = (val_relu - mean) * inv_std * w + b;
    }
}

torch::Tensor group_norm_relu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int64_t G, 
    double eps) {
    
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    auto output = torch::empty_like(input);
    
    const int threads = 512;
    const int blocks = N * G;
    size_t shared_mem_size = 2 * threads * input.element_size();
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "group_norm_relu_cuda", ([&] {
        group_norm_relu_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, D, H, W, G, static_cast<scalar_t>(eps)
        );
    }));
    
    return output;
}
"""

fused_gn_relu_cpp_source = "torch::Tensor group_norm_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int64_t G, double eps);"

# Compile the inline CUDA code
fused_gn_relu = load_inline(
    name="fused_gn_relu",
    cpp_sources=fused_gn_relu_cpp_source,
    cuda_sources=fused_gn_relu_source,
    functions=["group_norm_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        # GroupNorm parameters (affine=True by default in original model)
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.groups = groups
        self.eps = 1e-5
        self.fused_gn_relu = fused_gn_relu

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused ReLU + GroupNorm
        x = self.fused_gn_relu.group_norm_relu_cuda(x, self.weight, self.bias, self.groups, self.eps)
        return x