import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused LayerNorm, AvgPool3d, and GELU
# Optimization strategy:
# 1. Parallelize LayerNorm: Use block reduction to compute mean and variance.
#    Each block handles one (N, C) instance. Threads in the block cooperate to sum over D*H*W.
# 2. Fuse Pooling and GELU: After normalization, each thread computes its corresponding output pixel
#    by averaging a local window of the normalized data and applying GELU.
# 3. Memory Access: Use __restrict__ pointers and ensure coalesced access where possible.
#    LayerNorm reads are strided, but necessary for the operation.
# 4. Avoid recomputation: Mean and variance are computed once per (N, C) block and shared.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants for reduction
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Device function for warp reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel for fused LayerNorm + AvgPool3d + GELU
__global__ void fused_ln_pool_gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int D, int H, int W,
    int pool_D, int pool_H, int pool_W,
    float epsilon) {

    // Each block processes one (n, c) instance
    // Grid dimension: N * C
    int nc_idx = blockIdx.x;
    int n = nc_idx / C;
    int c = nc_idx % C;

    int spatial_size = D * H * W;
    
    // Pointers for this specific (n, c)
    const float* __restrict__ input_nc = input + nc_idx * spatial_size;
    
    // 1. Compute Mean and Variance using Block Reduction
    float sum_val = 0.0f;
    float sq_sum_val = 0.0f;

    // Stride loop to cover the entire spatial dimension
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = input_nc[i];
        sum_val += val;
        sq_sum_val += val * val;
    }

    // Reduce within warps
    sum_val = warp_reduce_sum(sum_val);
    sq_sum_val = warp_reduce_sum(sq_sum_val);

    // Use shared memory for the partial sums from each warp
    __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_sq_sum[BLOCK_SIZE / WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        s_sum[wid] = sum_val;
        s_sq_sum[wid] = sq_sum_val;
    }
    __syncthreads();

    // First warp performs the final reduction
    if (wid == 0) {
        sum_val = (lane < (BLOCK_SIZE / WARP_SIZE)) ? s_sum[lane] : 0.0f;
        sq_sum_val = (lane < (BLOCK_SIZE / WARP_SIZE)) ? s_sq_sum[lane] : 0.0f;
        sum_val = warp_reduce_sum(sum_val);
        sq_sum_val = warp_reduce_sum(sq_sum_val);

        // Broadcast result to all threads in the block via shared memory
        if (threadIdx.x == 0) {
            s_sum[0] = sum_val;
            s_sq_sum[0] = sq_sum_val;
        }
    }
    __syncthreads();

    float mean = s_sum[0] / spatial_size;
    float var = s_sq_sum[0] / spatial_size - mean * mean;
    float inv_std = rsqrtf(var + epsilon);
    
    float gamma_val = gamma[c];
    float beta_val = beta[c];

    // 2. Apply LayerNorm, Pooling, and GELU
    // Each thread computes one output pixel (od, oh, ow)
    // Output dimensions
    int out_D = (D + pool_D - 1) / pool_D;
    int out_H = (H + pool_H - 1) / pool_H;
    int out_W = (W + pool_W - 1) / pool_W;
    int out_spatial_size = out_D * out_H * out_W;

    // Iterate over output pixels assigned to this thread
    for (int out_idx = threadIdx.x; out_idx < out_spatial_size; out_idx += blockDim.x) {
        // Decompose output index
        int od = out_idx / (out_H * out_W);
        int tmp = out_idx % (out_H * out_W);
        int oh = tmp / out_W;
        int ow = tmp % out_W;

        // Calculate input pooling window boundaries
        int id_start = od * pool_D;
        int id_end = min(id_start + pool_D, D);
        int ih_start = oh * pool_H;
        int ih_end = min(ih_start + pool_H, H);
        int iw_start = ow * pool_W;
        int iw_end = min(iw_start + pool_W, W);

        float pool_sum = 0.0f;
        int pool_count = 0;

        // Pooling loop
        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    int in_idx = id * H * W + ih * W + iw;
                    float val = input_nc[in_idx];
                    
                    // LayerNorm
                    val = (val - mean) * inv_std * gamma_val + beta_val;
                    
                    pool_sum += val;
                    pool_count++;
                }
            }
        }

        float pooled_val = pool_sum / pool_count;
        
        // GELU (approximation)
        float x = pooled_val;
        float gelu = x * 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        
        // Write output
        // Output layout: (N, C, out_D, out_H, out_W)
        // Global index: (n, c, od, oh, ow)
        // We map the block to (n, c), so we just need to write to the correct offset
        output[nc_idx * out_spatial_size + out_idx] = gelu;
    }
}

torch::Tensor fused_ln_pool_gelu_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int pool_D, int pool_H, int pool_W,
    float epsilon) {

    auto N = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    int out_D = (D + pool_D - 1) / pool_D;
    int out_H = (H + pool_H - 1) / pool_H;
    int out_W = (W + pool_W - 1) / pool_W;

    auto output = torch::zeros({N, C, out_D, out_H, out_W}, input.options());

    // Launch configuration:
    // Grid: N * C blocks (one per instance)
    // Block: BLOCK_SIZE threads (to saturate reduction and output loop)
    int num_blocks = N * C;
    
    // Ensure we don't launch too many threads if spatial size is small, 
    // but BLOCK_SIZE is reasonable for reduction.
    fused_ln_pool_gelu_kernel<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, C, D, H, W,
        pool_D, pool_H, pool_W,
        epsilon
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ln_pool_gelu_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int pool_D, int pool_H, int pool_W, float epsilon);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ln_pool_gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by a sum, 
    and a fused kernel for LayerNorm, Average Pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        
        # We need to access the weights of LayerNorm directly for the custom kernel
        # norm_shape is expected to be (out_channels,) based on the problem description
        self.norm = nn.LayerNorm(norm_shape)
        
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Transposed Convolution (using PyTorch's optimized implementation)
        x = self.conv_transpose(x)
        
        # 2. Sum (Element-wise addition)
        # Broadcasting sum_weight (scalar) across the tensor
        x = x + self.sum_weight
        
        # 3. Fused LayerNorm + AvgPool + GELU
        # Extract parameters for the custom kernel
        gamma = self.norm.weight
        beta = self.norm.bias
        epsilon = self.norm.eps
        
        # Call the custom CUDA kernel
        x = self.fused_ops.fused_ln_pool_gelu_cuda(
            x, gamma, beta, 
            self.pool_kernel_size[0], 
            self.pool_kernel_size[1], 
            self.pool_kernel_size[2], 
            epsilon
        )
        
        return x