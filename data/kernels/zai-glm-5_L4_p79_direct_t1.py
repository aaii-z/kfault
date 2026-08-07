import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for Instance Normalization fused with element-wise multiplication and clamping
// Normalization is performed per channel per sample.
// x_norm = (x - mean) / sqrt(var + eps)
// out = clamp(x_norm * multiplier, min, max)
__global__ void fused_norm_mul_clamp_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ multiplier, 
    float* __restrict__ out,
    int N, int C, int D, int H, int W, 
    float clamp_min, float clamp_max, float eps) {
    
    // Each block processes one (n, c) instance
    int n = blockIdx.x / C;
    int c = blockIdx.x % C;
    
    int spatial_size = D * H * W;
    int offset = (n * C + c) * spatial_size;
    
    // Shared memory for reduction (sum and sum of squares)
    // We use a struct to handle the two values
    struct SumSq { float sum; float sq; };
    extern __shared__ char shared_mem_char[];
    SumSq* sdata = (SumSq*)shared_mem_char;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Grid stride loop to compute sum and sum_sq
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int idx = offset + i;
        float val = x[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    sdata[threadIdx.x].sum = local_sum;
    sdata[threadIdx.x].sq = local_sum_sq;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x].sum += sdata[threadIdx.x + s].sum;
            sdata[threadIdx.x].sq += sdata[threadIdx.x + s].sq;
        }
        __syncthreads();
    }
    
    float mean = sdata[0].sum / spatial_size;
    float var = sdata[0].sq / spatial_size - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    float mul_val = multiplier[c]; // Multiplier shape (C, 1, 1, 1)
    
    // Apply normalization, multiplication, and clamping
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int idx = offset + i;
        float val = x[idx];
        float norm_val = (val - mean) * inv_std;
        float res = norm_val * mul_val;
        // Clamp
        res = fmaxf(clamp_min, fminf(clamp_max, res));
        out[idx] = res;
    }
}

// Kernel for Max reduction along dimension 1 (Channels)
// Input: (N, C, D, H, W) -> Output: (N, D, H, W)
__global__ void max_reduce_kernel(
    const float* __restrict__ x, 
    float* __restrict__ out, 
    int N, int C, int D, int H, int W) {
    
    int n = blockIdx.x;
    int spatial_idx = blockIdx.y; // 0 to D*H*W-1
    
    int d = spatial_idx / (H * W);
    int hw = spatial_idx % (H * W);
    int h = hw / W;
    int w = hw % W;
    
    int base_idx = n * C * D * H * W + d * H * W + h * W + w;
    
    float max_val = -INFINITY;
    
    for (int c = 0; c < C; ++c) {
        float val = x[base_idx + c * D * H * W];
        if (val > max_val) {
            max_val = val;
        }
    }
    
    out[n * D * H * W + spatial_idx] = max_val;
}

// C++ interface
torch::Tensor fused_ops_cuda(
    torch::Tensor x, 
    torch::Tensor multiplier, 
    torch::Tensor conv_weight, 
    torch::Tensor conv_bias, 
    int kernel_size, 
    float clamp_min, float clamp_max) {
    
    // 1. Conv3D
    // We use at::conv3d for simplicity and performance as writing a highly optimized Conv3D is complex
    // and usually cuDNN backed implementations are hard to beat.
    // The main optimization focus here is the fusion of the subsequent ops.
    // Options: padding=(kernel_size-1)/2, stride=1 (default)
    int padding = (kernel_size - 1) / 2;
    auto x_conv = at::conv3d(x, conv_weight, conv_bias, /*stride=*/1, /*padding=*/padding);
    
    // 2. Fused Instance Norm + Mul + Clamp
    auto sizes = x_conv.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    auto out_norm = torch::empty_like(x_conv);
    
    const int threads = 256;
    const int blocks = N * C; // One block per channel per batch item
    // Shared memory size: 2 floats (sum, sum_sq) per thread
    size_t shared_mem_size = threads * 2 * sizeof(float);
    
    fused_norm_mul_clamp_kernel<<<blocks, threads, shared_mem_size>>>(
        x_conv.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        out_norm.data_ptr<float>(),
        N, C, D, H, W,
        clamp_min, clamp_max, 1e-5f
    );
    
    // 3. Max reduction
    // We need to reduce along dim 1.
    // Output shape: (N, D, H, W)
    auto out_max = torch::empty({N, D, H, W}, x_conv.options());
    
    // Grid: N blocks for batch, D*H*W blocks for spatial dimensions? 
    // Or N * D*H*W blocks. Let's map one block per output element.
    // Grid dim: (N, D*H*W)
    dim3 grid(N, D * H * W);
    
    max_reduce_kernel<<<grid, threads>>>(
        out_norm.data_ptr<float>(),
        out_max.data_ptr<float>(),
        N, C, D, H, W
    );
    
    return out_max;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor multiplier, torch::Tensor conv_weight, torch::Tensor conv_bias, int kernel_size, float clamp_min, float clamp_max);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model using custom CUDA kernels.
    Fuses InstanceNorm, Multiplication, Clamping, and Max reduction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # We keep the Conv3d weights as Parameters but handle the forward pass manually
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_ops = fused_ops

    def forward(self, x):
        # Call the custom fused CUDA kernel
        # We pass the weights and bias directly
        return self.fused_ops.fused_ops_cuda(
            x, 
            self.multiplier, 
            self.conv.weight, 
            self.conv.bias, 
            self.conv.kernel_size[0], 
            self.clamp_min, 
            self.clamp_max
        )