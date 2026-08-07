import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: Min, Sum, GELU, Add
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Approximation for GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_ops_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    // Each thread handles one element in the output tensor (N, 1, 1, W)
    // Output dimensions are (N, 1, 1, W) based on reduction logic
    
    int nw = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * W;
    
    if (nw < total_size) {
        int n = nw / W;
        int w = nw % W;
        
        float min_val = INFINITY;
        
        // Iterate over Channels (C) and Height (H) to perform Min and Sum
        // Min over C, Sum over H
        // We can fuse these loops: min_val = sum_h( min_c( x[n, c, h, w] ) )
        
        // Actually, looking at the PyTorch code:
        // x = torch.min(x, dim=1, keepdim=True)[0] -> Shape (N, 1, H, W)
        // x = torch.sum(x, dim=2, keepdim=True)    -> Shape (N, 1, 1, W)
        
        // So for a specific (n, w), we need to:
        // 1. Find min along C for each h. Let's call this m_h.
        // 2. Sum m_h over all h.
        
        float sum_val = 0.0f;
        
        for (int h = 0; h < H; ++h) {
            float min_c = INFINITY;
            for (int c = 0; c < C; ++c) {
                // Input tensor layout is (N, C, H, W)
                // Linear index: n * C * H * W + c * H * W + h * W + w
                int idx = n * C * H * W + c * H * W + h * W + w;
                float val = input[idx];
                if (val < min_c) {
                    min_c = val;
                }
            }
            sum_val += min_c;
        }
        
        // Apply GELU
        float gelu_val = gelu_approx(sum_val);
        
        // Add bias
        // Bias shape is (1, 1, 1), so bias[0]
        float result = gelu_val + bias[0];
        
        // Write output
        // Output layout (N, 1, 1, W)
        output[nw] = result;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias) {
    // Input shape: (N, C, H, W)
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    
    // Output shape: (N, 1, 1, W)
    auto output = torch::zeros({N, 1, 1, W}, input.options());
    
    const int block_size = 256;
    int total_elements = N * W;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias);"
)

# Compile the inline CUDA code
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
    Optimized model using custom CUDA kernels.
    Replaces ConvTranspose2d with a standard PyTorch layer, but fuses
    Min, Sum, GELU, and Add operations into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        # Standard ConvTranspose2d
        x = self.conv_transpose(x)
        
        # Fused CUDA kernel for Min, Sum, GELU, Add
        # The kernel expects input (N, C, H, W) and bias
        x = self.fused_ops.fused_ops_cuda(x, self.bias)
        
        return x