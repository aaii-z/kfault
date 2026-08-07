import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: subtract1 -> tanh -> subtract2 -> avg_pool
__global__ void fused_ops_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    float subtract1_val, float subtract2_val,
    int kernel_size_pool) {

    int H_out = H_in / kernel_size_pool;
    int W_out = W_in / kernel_size_pool;
    
    // Calculate output pixel index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;
    
    if (idx < total_elements) {
        // Decompose idx into (n, c, h_out, w_out)
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);
        
        // Calculate input region start
        int h_in_start = h_out * kernel_size_pool;
        int w_in_start = w_out * kernel_size_pool;
        
        float sum = 0.0f;
        float pool_area = static_cast<float>(kernel_size_pool * kernel_size_pool);
        
        // Input tensor strides: (N, C, H, W)
        int input_offset = ((n * C + c) * H_in + h_in_start) * W_in + w_in_start;
        
        // Loop over the pooling window
        for (int kh = 0; kh < kernel_size_pool; ++kh) {
            for (int kw = 0; kw < kernel_size_pool; ++kw) {
                // Calculate linear index for input
                int in_idx = input_offset + kh * W_in + kw;
                
                // Chain of operations
                float val = input[in_idx];
                val = val - subtract1_val;
                val = tanhf(val);
                val = val - subtract2_val;
                
                sum += val;
            }
        }
        
        output[idx] = sum / pool_area;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    float subtract1_val,
    float subtract2_val,
    int kernel_size_pool) {

    // Input dimensions
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Output dimensions
    int H_out = H_in / kernel_size_pool;
    int W_out = W_in / kernel_size_pool;
    
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());
    
    int total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        subtract1_val, subtract2_val,
        kernel_size_pool);
        
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, float subtract1_val, float subtract2_val, int kernel_size_pool);"
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
    Optimized model that performs a convolution followed by a fused CUDA kernel
    for subtract, tanh, subtract, and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_ops_cuda(x, self.subtract1_value, self.subtract2_value, self.kernel_size_pool)
        return x