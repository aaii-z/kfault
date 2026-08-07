import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__global__ void fused_min_sum_gelu_add_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int N, int C, int H, int W) {
    
    // Input shape: (N, C, H, W)
    // Output shape: (N, 1, 1, W)
    // Bias shape: (1, 1, 1)
    
    int n = blockIdx.y;
    int w_block_idx = blockIdx.x;
    
    const int TILE_W = 16;
    const int TILE_H = 16;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int w = w_block_idx * TILE_W + tx;
    
    // Shared memory to reduce across H (ty dimension)
    // Size TILE_W * TILE_H
    extern __shared__ float smem[];
    
    float sum_val = 0.0f;
    
    // Iterate over H
    // Each thread (ty) handles a subset of H
    if (w < W) {
        for (int h = ty; h < H; h += TILE_H) {
            float min_val = FLT_MAX;
            
            // Base pointer for this (n, h, w)
            // Input layout is (N, C, H, W)
            // We iterate over C
            // Optimization: Calculate base offset for (n, h, w) and stride by C*H*W
            
            // Actually, input is (N, C, H, W).
            // Offset for (n, c, h, w) is n*C*H*W + c*H*W + h*W + w.
            // For fixed n, h, w, we iterate c.
            // Stride is C*H*W? No.
            // n*C*H*W is base for batch n.
            // h*W + w is offset inside spatial.
            // We iterate c. Stride is H*W.
            
            const float* ptr_n = input + n * C * H * W;
            // Iterate over channels
            for (int c = 0; c < C; ++c) {
                float val = ptr_n[c * H * W + h * W + w];
                if (val < min_val) {
                    min_val = val;
                }
            }
            sum_val += min_val;
        }
    }
    
    // Write partial sum to shared memory
    // smem[ty * TILE_W + tx]
    smem[ty * TILE_W + tx] = sum_val;
    __syncthreads();
    
    // Reduction in shared memory
    // Sum along ty dimension
    for (int s = TILE_H / 2; s > 0; s >>= 1) {
        if (ty < s) {
            smem[ty * TILE_W + tx] += smem[(ty + s) * TILE_W + tx];
        }
        __syncthreads();
    }
    
    // Thread (tx, 0) writes the result
    if (ty == 0 && w < W) {
        float val = smem[tx];
        
        // GELU activation
        // Exact formula: x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
        float sqrt2 = 1.41421356237f;
        float gelu_val = val * 0.5f * (1.0f + erff(val / sqrt2));
        
        // Add bias
        gelu_val += bias[0];
        
        // Write output
        // Output shape (N, 1, 1, W)
        output[n * W + w] = gelu_val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias) {
    // Input: (N, C, H, W)
    // Output: (N, 1, 1, W)
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    auto output = torch::zeros({N, 1, 1, W}, input.options());
    
    // Config
    const int TILE_W = 16;
    const int TILE_H = 16;
    
    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W, N);
    
    size_t shared_mem_size = TILE_W * TILE_H * sizeof(float);
    
    fused_min_sum_gelu_add_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    Optimized with a custom CUDA kernel fusing min, sum, gelu, and add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        # Use the custom fused CUDA kernel
        x = self.fused_op.fused_op_cuda(x, self.bias)
        return x