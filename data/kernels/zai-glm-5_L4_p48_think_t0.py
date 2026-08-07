import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: mul, tanh, mul, sigmoid
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_ops_kernel(const float* __restrict__ input, 
                                 const float* __restrict__ scale, 
                                 const float* __restrict__ bias, 
                                 float* __restrict__ output, 
                                 int n_elements, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        // Calculate channel index (c) based on the NCDHW layout
        // idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w
        // We need c to index scale and bias
        int c_idx = (idx / (D * H * W)) % C;
        
        float val = input[idx];
        float s = scale[c_idx];
        float b = bias[c_idx];
        
        // Operation 1: x * scaling_factor
        val = val * s;
        
        // Operation 2: tanh
        val = tanhf(val);
        
        // Operation 3: x * bias
        val = val * b;
        
        // Operation 4: sigmoid
        val = 1.0f / (1.0f + expf(-val));
        
        output[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {
    // Ensure inputs are contiguous for simple pointer arithmetic
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "Scale tensor must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias tensor must be contiguous");
    
    auto output = torch::empty_like(input);
    int n_elements = input.numel();
    
    // Extract dimensions for index calculation
    // Input shape: (N, C, D, H, W)
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    const int block_size = 256;
    const int num_blocks = (n_elements + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements,
        C, D, H, W
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution followed by a fused custom CUDA kernel
    for scaling, tanh, bias multiplication, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # Note: The argument 'scaling_factor' is unused in the original architecture's initialization logic,
        # as it creates a random parameter of shape bias_shape. We preserve this behavior.
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # Apply the fused custom CUDA kernel
        x = self.fused_ops.fused_ops_cuda(x, self.scaling_factor, self.bias)
        return x