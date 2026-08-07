import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused softmax, bias, scale, and sigmoid
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

// Kernel for fused operations: Softmax (dim=1) + Bias + Scale + Sigmoid
// Optimized for NCHW memory layout.
// Uses Online Softmax algorithm for numerical stability and single-pass reduction.
// Each thread processes one spatial pixel (h, w) across all channels.

__global__ void fused_ops_kernel(const float* __restrict__ input, 
                                 const float* __restrict__ bias, 
                                 float* __restrict__ output, 
                                 int N, int C, int H, int W, 
                                 float scaling_factor) {
    // Calculate spatial coordinates
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int n_idx = blockIdx.z;

    if (n_idx < N && h_idx < H && w_idx < W) {
        // Online Softmax variables
        float max_val = -std::numeric_limits<float>::infinity();
        float sum_exp = 0.0f;

        // Base offset for the current pixel in the NCHW tensor
        // Input shape: (N, C, H, W)
        // Offset = n * C * H * W + h * W + w
        // We iterate over c, so the stride for c is H * W
        int base_offset = n_idx * C * H * W + h_idx * W + w_idx;
        int channel_stride = H * W;

        // Pass 1: Compute Max and Sum using Online Softmax algorithm
        for (int c = 0; c < C; ++c) {
            float val = input[base_offset + c * channel_stride];
            
            // Online softmax update rule
            float new_max = fmaxf(max_val, val);
            float correction = expf(max_val - new_max);
            float term = expf(val - new_max);
            
            sum_exp = sum_exp * correction + term;
            max_val = new_max;
        }

        // Pass 2: Normalize and apply post-softmax operations
        for (int c = 0; c < C; ++c) {
            int idx = base_offset + c * channel_stride;
            float val = input[idx];
            
            // Softmax
            float softmax_val = expf(val - max_val) / sum_exp;
            
            // Add bias (bias shape is (C, 1, 1), effectively a 1D tensor of size C)
            float biased_val = softmax_val + bias[c];
            
            // Scale
            float scaled_val = biased_val * scaling_factor;
            
            // Sigmoid
            float sig_val = 1.0f / (1.0f + expf(-scaled_val));
            
            output[idx] = sig_val;
        }
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor) {
    // Ensure input is contiguous
    input = input.contiguous();
    
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::empty_like(input);
    
    // Block and Grid configuration
    // Using 2D blocks for spatial dimensions, grid depth for batch dimension
    dim3 block(16, 16); 
    dim3 grid((W + block.x - 1) / block.x, 
              (H + block.y - 1) / block.y, 
              N);
    
    fused_ops_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        scaling_factor
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"
)

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
    Optimized Model that performs a transposed convolution followed by a fused
    softmax, bias addition, scaling, and sigmoid operation using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        # Standard transposed convolution
        x = self.conv_transpose(x)
        
        # Fused operation: Softmax + Bias + Scale + Sigmoid
        # The custom kernel handles the entire post-conv pipeline
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.scaling_factor)
        
        return x