import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: Mul + LeakyReLU + GELU
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ multiplier, 
    float* __restrict__ output, 
    int num_elements, 
    int channels, int hw, 
    float negative_slope) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        // Calculate channel index for broadcasting
        // Assuming NCHW layout: input shape (N, C, H, W)
        // i = n * C * H * W + c * H * W + spatial_idx
        // c = (i / hw) % channels
        int c = (i / hw) % channels;
        
        float val = input[i];
        float mult = multiplier[c];
        
        // 1. Multiplication
        val *= mult;
        
        // 2. LeakyReLU
        val = (val > 0.0f) ? val : val * negative_slope;
        
        // 3. GELU (tanh approximation)
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // sqrt(2/pi) approx 0.7978845608
        float k = 0.7978845608028654f;
        float x_cubed = val * val * val;
        float inner = k * (val + 0.044715f * x_cubed);
        val = 0.5f * val * (1.0f + tanhf(inner));
        
        output[i] = val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor multiplier, float negative_slope) {
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    int num_elements = input.numel();
    int hw = H * W;

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_op_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        multiplier.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements, 
        C, hw, 
        negative_slope
    );

    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor multiplier, float negative_slope);"

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
    Optimized Model that performs a convolution, followed by a fused custom CUDA kernel
    for multiplication, LeakyReLU, and GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self.negative_slope = 0.01 # Default for LeakyReLU
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        # Apply fused operation: x * multiplier -> LeakyReLU -> GELU
        x = self.fused_op.fused_op_cuda(x, self.multiplier, self.negative_slope)
        return x