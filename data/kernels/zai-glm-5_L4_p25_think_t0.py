import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Min, Tanh, Tanh
fused_min_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_min_tanh_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                      int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = B * H * W;
    
    if (idx < total_pixels) {
        // Map flat index to (b, h, w) coordinates for the output tensor (B, 1, H, W)
        int b = idx / (H * W);
        int rem = idx % (H * W);
        int h = rem / W;
        int w = rem % W;
        
        // Calculate base offset for the input tensor (B, C, H, W)
        // Input is contiguous: offset = b * (C*H*W) + c * (H*W) + h * W + w
        long long base_offset = (long long)b * C * H * W + (long long)h * W + w;
        long long channel_stride = (long long)H * W;
        
        // Initialize min_val with the first channel
        float min_val = input[base_offset];
        
        // Iterate over channels to find the minimum
        for (int c = 1; c < C; ++c) {
            float val = input[base_offset + c * channel_stride];
            if (val < min_val) {
                min_val = val;
            }
        }
        
        // Apply Tanh twice
        float res = tanhf(min_val);
        res = tanhf(res);
        
        output[idx] = res;
    }
}

torch::Tensor fused_min_tanh_cuda(torch::Tensor input) {
    // Input shape: B, C, H, W
    auto sizes = input.sizes();
    int B = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    
    // Output shape: B, 1, H, W
    auto output = torch::zeros({B, 1, H, W}, input.options());
    
    const int block_size = 256;
    int total_pixels = B * H * W;
    const int num_blocks = (total_pixels + block_size - 1) / block_size;
    
    fused_min_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        B, C, H, W
    );
    
    return output;
}
"""

fused_min_tanh_cpp_source = (
    "torch::Tensor fused_min_tanh_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_min_tanh_op = load_inline(
    name="fused_min_tanh",
    cpp_sources=fused_min_tanh_cpp_source,
    cuda_sources=fused_min_tanh_source,
    functions=["fused_min_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution followed by a fused Min-Tanh-Tanh operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_op = fused_min_tanh_op

    def forward(self, x):
        x = self.conv(x)
        # Use the custom fused CUDA kernel
        x = self.fused_op.fused_min_tanh_cuda(x)
        return x