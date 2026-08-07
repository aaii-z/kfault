import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void fused_op_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int B, int C, int D, int H, int W,
    scalar_t scaling_factor) {

    // Block indices
    int tx = threadIdx.x; // 0..15 (W dimension tile)
    int ty = threadIdx.y; // 0..63 (C dimension)
    
    // Grid indices
    int w_idx = blockIdx.x * 16 + tx;
    int h_idx = blockIdx.y;
    int b_idx = blockIdx.z;

    if (w_idx >= W) return;

    // Shared memory for reduction
    // Layout: [C][W_tile] -> [64][17] (padding to avoid bank conflicts)
    __shared__ scalar_t s_data[64][17];

    // Register for value
    scalar_t val_reg = static_cast<scalar_t>(0.0);

    // 1. Mean over D and Add Bias
    if (ty < C) {
        // Input pointer arithmetic
        // Input shape: (B, C, D, H, W)
        // Strides: C*D*H*W, D*H*W, H*W, W, 1
        int base_idx = b_idx * (C * D * H * W) + ty * (D * H * W) + h_idx * W + w_idx;
        const scalar_t* in_ptr = input + base_idx;
        
        scalar_t sum_d = 0.0;
        for (int d = 0; d < D; ++d) {
            sum_d += in_ptr[d * H * W];
        }
        
        val_reg = sum_d / static_cast<scalar_t>(D);
        val_reg += bias[ty]; // Bias is (C)
    }
    
    // 2. Max Reduction (Softmax part 1)
    // Load to shared memory
    if (ty < C) {
        s_data[ty][tx] = val_reg;
    } else {
        s_data[ty][tx] = -INFINITY;
    }
    __syncthreads();

    // Reduce over ty (dim 0 of s_data)
    for (int s = 32; s > 0; s >>= 1) {
        if (ty < s) {
            scalar_t val1 = s_data[ty][tx];
            scalar_t val2 = s_data[ty + s][tx];
            s_data[ty][tx] = fmax(val1, val2);
        }
        __syncthreads();
    }
    
    scalar_t max_val = s_data[0][tx];
    __syncthreads();

    // 3. Exp and Sum Reduction (Softmax part 2)
    scalar_t exp_reg = 0.0;
    if (ty < C) {
        exp_reg = exp(val_reg - max_val);
        s_data[ty][tx] = exp_reg;
    } else {
        s_data[ty][tx] = 0.0;
    }
    __syncthreads();

    for (int s = 32; s > 0; s >>= 1) {
        if (ty < s) {
            s_data[ty][tx] += s_data[ty + s][tx];
        }
        __syncthreads();
    }
    
    scalar_t sum_val = s_data[0][tx];
    
    // 4. Finalize: Softmax, Tanh, Scale
    if (ty < C) {
        scalar_t softmax_val = exp_reg / sum_val;
        scalar_t tanh_val = tanh(softmax_val);
        scalar_t out_val = tanh_val * scaling_factor;
        
        // Write output
        // Output shape: (B, C, 1, H, W)
        // Strides: C*H*W, H*W, H*W, W, 1
        output[b_idx * (C * H * W) + ty * (H * W) + h_idx * W + w_idx] = out_val;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, 
    torch::Tensor bias, 
    float scaling_factor) {
    
    auto sizes = input.sizes();
    int B = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    // Output shape: (B, C, 1, H, W)
    auto output = torch::zeros({B, C, 1, H, W}, input.options());

    // Grid: (W_tiles, H, B)
    // Block: (16, 64) -> 16 for W, 64 for C
    dim3 block(16, 64);
    dim3 grid((W + 15) / 16, H, B);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_op_kernel", ([&] {
        fused_op_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, C, D, H, W,
            static_cast<scalar_t>(scaling_factor)
        );
    }));

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);
"""

# Compile the inline CUDA code
fused_op_module = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op_module

    def forward(self, x):
        x = self.conv_transpose(x)
        # Use custom fused kernel
        # Note: The bias in the original model is (1, C, 1, 1, 1). 
        # The kernel expects a 1D bias tensor of size C for efficient access.
        bias_1d = self.bias.squeeze() # Shape (C)
        x = self.fused_op.fused_op_cuda(x, bias_1d, self.scaling_factor)
        return x