import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + Tanh + HardSwish + Residual Add + LogSumExp
# Strategy:
# 1. Kernel 1 (Fused Post-Conv): Fuses GroupNorm, Tanh, HardSwish, and Residual Addition.
#    It takes the convolution output (x_conv) as input.
#    It computes GroupNorm stats, then applies the sequence of ops.
#    It outputs the residual result (x_res).
#    This saves memory bandwidth by avoiding intermediate writes to global memory for norm, tanh, and hardswish outputs.
# 2. Kernel 2 (LogSumExp): Computes LogSumExp over the channel dimension.
#    Uses a tiled reduction approach where each block handles a spatial tile (16x16 pixels)
#    and iterates over channels. This ensures coalesced global memory access.

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants
#define TILE_SIZE 16

// Kernel 1: Fused GroupNorm + Tanh + HardSwish + Residual Add
// Grid: (N * G), Block: 256
// Each block handles one group in one sample.
__global__ void fused_gn_tanh_hs_add_kernel(
    const float* __restrict__ x_conv,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int H, int W, int G, float eps) {

    int block_idx = blockIdx.x;
    int n = block_idx / G;
    int g = block_idx % G;

    int c_per_g = C / G;
    int group_size = H * W * c_per_g;
    
    // Pointers to the start of the data for this sample and group
    // x_conv shape: (N, C, H, W)
    // Group g corresponds to channels [g * c_per_g, (g+1) * c_per_g)
    const float* x_ptr = x_conv + n * C * H * W + g * c_per_g * H * W;
    float* out_ptr = out + n * C * H * W + g * c_per_g * H * W;
    const float* w_ptr = weight + g * c_per_g;
    const float* b_ptr = bias + g * c_per_g;

    // Shared memory for reduction (sum and sum of squares)
    // Using a struct or separate arrays
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    __shared__ float s_mean[1];
    __shared__ float s_var[1];

    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Pass 1: Compute mean and variance
    // Grid-stride loop over the elements belonging to this group
    for (int i = tid; i < group_size; i += blockDim.x) {
        float val = x_ptr[i];
        local_sum += val;
        local_sum_sq += val * val;
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

    if (tid == 0) {
        float mean = s_sum[0] / group_size;
        float var = s_sum_sq[0] / group_size - mean * mean;
        s_mean[0] = mean;
        s_var[0] = var;
    }
    __syncthreads();

    float mean = s_mean[0];
    float var = s_var[0];
    float inv_std = rsqrtf(var + eps);

    // Pass 2: Normalize, Activate, and Add
    for (int i = tid; i < group_size; i += blockDim.x) {
        // Calculate channel index to fetch weight and bias
        // Linear index i within the group corresponds to (c_local, y, x)
        // i = c_local * (H * W) + y * W + x
        int c_local = i / (H * W);
        
        float val = x_ptr[i];
        
        // Group Norm
        float norm = (val - mean) * inv_std;
        float w = w_ptr[c_local];
        float b = b_ptr[c_local];
        norm = norm * w + b;
        
        // Tanh
        float t = tanhf(norm);
        
        // HardSwish: x * relu6(x + 3) / 6
        // relu6(x) = min(max(x, 0), 6)
        float hs_arg = t + 3.0f;
        hs_arg = fmaxf(0.0f, hs_arg);
        hs_arg = fminf(6.0f, hs_arg);
        float hs = t * hs_arg / 6.0f;
        
        // Residual Add
        out_ptr[i] = val + hs;
    }
}

// Kernel 2: LogSumExp over channels
// Grid: (N, H/16, W/16), Block: 16x16
// Each thread handles one pixel (x, y) across all channels.
// Reduction is done in registers (sequential loop over channels).
__global__ void logsumexp_kernel(
    const float* __restrict__ x_res,
    float* __restrict__ out,
    int N, int C, int H, int W) {

    int w_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
    int h_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int n_idx = blockIdx.z;

    if (w_idx < W && h_idx < H) {
        // Base pointer for this spatial location across all channels
        // Input is (N, C, H, W)
        // Offset for (n, 0, h, w) is n*C*H*W + h*W + w
        int base_offset = n_idx * C * H * W + h_idx * W + w_idx;
        int stride = H * W; // Stride to jump to next channel

        float max_val = -INFINITY;
        float sum_exp = 0.0f;

        // Pass 1: Find max
        for (int c = 0; c < C; ++c) {
            float val = x_res[base_offset + c * stride];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Pass 2: Sum exp(val - max)
        for (int c = 0; c < C; ++c) {
            float val = x_res[base_offset + c * stride];
            sum_exp += expf(val - max_val);
        }

        // Result
        float result = logf(sum_exp) + max_val;
        
        // Output is (N, 1, H, W)
        // Offset: n*H*W + h*W + w
        out[n_idx * H * W + h_idx * W + w_idx] = result;
    }
}

// C++ interface
torch::Tensor forward_cuda(
    torch::Tensor x_conv, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int G, float eps) {

    int N = x_conv.size(0);
    int C = x_conv.size(1);
    int H = x_conv.size(2);
    int W = x_conv.size(3);

    // Allocate intermediate tensor for residual result
    auto x_res = torch::empty_like(x_conv);

    // Launch Kernel 1
    int blocks = N * G;
    int threads = 256;
    fused_gn_tanh_hs_add_kernel<<<blocks, threads>>>(
        x_conv.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        x_res.data_ptr<float>(),
        N, C, H, W, G, eps
    );

    // Allocate output tensor for LogSumExp
    // Output shape: (N, 1, H, W)
    auto out = torch::empty({N, 1, H, W}, x_res.options());

    // Launch Kernel 2
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks((W + TILE_SIZE - 1) / TILE_SIZE, 
                    (H + TILE_SIZE - 1) / TILE_SIZE, 
                    N);
    
    logsumexp_kernel<<<num_blocks, threads_per_block>>>(
        x_res.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W
    );

    return out;
}
"""

cpp_source = """
torch::Tensor forward_cuda(torch::Tensor x_conv, torch::Tensor weight, torch::Tensor bias, int G, float eps);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.eps = eps
        self.groups = groups
        self.fused_ops = fused_ops

    def forward(self, x):
        # Step 1: Convolution (using PyTorch's optimized implementation)
        x_conv = self.conv(x)
        
        # Step 2: Fused GroupNorm + Tanh + HardSwish + Residual Add + LogSumExp
        # We use the weights from the original GroupNorm module
        return self.fused_ops.forward_cuda(
            x_conv, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.groups, 
            self.eps
        )