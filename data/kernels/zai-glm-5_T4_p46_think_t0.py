import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused subtraction, tanh, and average pooling
fused_sub_tanh_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_sub_tanh_sub_pool_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    float sub1, float sub2, 
    int pool_size,
    int N, int C, int H, int W) {
    
    // Output dimensions
    int H_out = H / pool_size;
    int W_out = W / pool_size;
    
    // Calculate output coordinates
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z; // Combined index for N and C

    if (w_out < W_out && h_out < H_out && nc < N * C) {
        int n = nc / C;
        int c = nc % C;
        
        int h_in_base = h_out * pool_size;
        int w_in_base = w_out * pool_size;
        
        float sum = 0.0f;
        
        // Loop over the pooling window
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                int h_in = h_in_base + i;
                int w_in = w_in_base + j;
                
                // Calculate input index (NCHW layout)
                long long idx = (long long)n * C * H * W + c * H * W + h_in * W + w_in;
                
                float val = input[idx];
                // Apply subtraction, tanh, subtraction
                val = val - sub1;
                val = tanhf(val);
                val = val - sub2;
                sum += val;
            }
        }
        
        // Calculate average and write to output
        float avg = sum / (pool_size * pool_size);
        output[(long long)n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out] = avg;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, 
    float sub1, float sub2, 
    int pool_size) {
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    int H_out = H / pool_size;
    int W_out = W / pool_size;
    
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());
    
    const int block_size_x = 16;
    const int block_size_y = 16;
    
    dim3 block(block_size_x, block_size_y);
    dim3 grid((W_out + block_size_x - 1) / block_size_x, 
              (H_out + block_size_y - 1) / block_size_y, 
              N * C);
              
    fused_sub_tanh_sub_pool_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        sub1, sub2, 
        pool_size,
        N, C, H, W);
        
    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, float sub1, float sub2, int pool_size);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_sub_tanh_pool_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution followed by a fused CUDA kernel
    for subtraction, tanh, subtraction, and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        # Use the custom fused CUDA kernel
        x = self.fused_op.fused_op_cuda(x, self.subtract1_value, self.subtract2_value, self.kernel_size_pool)
        return x