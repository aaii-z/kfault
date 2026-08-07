import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activations and bias addition
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function for approximate GELU (faster than exact GELU)
__device__ __forceinline__ float gelu_approx(float x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Using the approximation for speed
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_ops_kernel(const float* __restrict__ input, const float* __restrict__ bias, float* __restrict__ output, int size, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Calculate the channel index to apply the correct bias
        // Input layout is (N, C, D, H, W)
        // We need to find the 'C' index for the current linear index
        int n = idx / (channels * depth * height * width);
        int remainder = idx % (channels * depth * height * width);
        int c = remainder / (depth * height * width);
        
        float val = input[idx];
        
        // 1. ReLU
        val = fmaxf(0.0f, val);
        
        // 2. Leaky ReLU (negative_slope=0.01)
        val = (val > 0.0f) ? val : val * 0.01f;
        
        // 3. GELU (approximate)
        val = gelu_approx(val);
        
        // 4. Sigmoid
        val = 1.0f / (1.0f + expf(-val));
        
        // 5. Add bias
        val += bias[c];
        
        output[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // Get dimensions for bias indexing
    // input shape: [batch, channels, depth, height, width]
    auto sizes = input.sizes();
    int channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, 
        channels, 
        depth, 
        height, 
        width
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias);"
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
    Optimized model that performs a 3D convolution followed by a fused kernel for ReLU, LeakyReLU, GELU, Sigmoid, and bias addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias)
        return x