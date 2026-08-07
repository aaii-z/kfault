import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias add, clamp, scale, clamp, divide
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void fused_ops_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out, 
                                 int N, int C, int H, int W, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * H * W;
    
    if (idx < total_size) {
        // Calculate indices
        int n = idx / (C * H * W);
        int c = (idx % (C * H * W)) / (H * W);
        int hw = idx % (H * W);
        
        // Load input and bias
        float val = x[idx];
        float b = bias[c];
        
        // Fused operations: add bias, clamp(0,1), scale, clamp(0,1), divide
        val = val + b;
        val = fmaxf(0.0f, fminf(1.0f, val));
        val = val * scaling_factor;
        val = fmaxf(0.0f, fminf(1.0f, val));
        val = val / scaling_factor;
        
        out[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias, float scaling_factor) {
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (N * C * H * W + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, C, H, W, 
        scaling_factor
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code for fused operations
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution followed by fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.scaling_factor)
        return x