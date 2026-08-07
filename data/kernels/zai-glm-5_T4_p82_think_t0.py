import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Tanh, Scale, Bias, and MaxPool
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_tanh_scale_bias_maxpool_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int N, int C, int H_in, int W_in, 
    int H_out, int W_out, 
    float scaling_factor, int pool_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;
    
    if (idx < total_elements) {
        // Calculate output coordinates (n, c, h_out, w_out) from flattened index
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);
        
        // Calculate the top-left corner of the pooling window in the input tensor
        int h_in_start = h_out * pool_size;
        int w_in_start = w_out * pool_size;
        
        float max_val = -INFINITY; 
        
        // Iterate over the pooling window
        // We assume the window fits within input bounds based on PyTorch MaxPool2d logic
        for (int kh = 0; kh < pool_size; ++kh) {
            for (int kw = 0; kw < pool_size; ++kw) {
                int h_in = h_in_start + kh;
                int w_in = w_in_start + kw;
                
                // Calculate input index (NCHW format)
                long long input_idx = (long long)n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                
                float val = input[input_idx];
                
                // Fused operations: tanh -> scale -> bias
                val = tanhf(val);
                val = val * scaling_factor;
                val = val + bias[c];
                
                // Max pooling reduction
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor fused_tanh_scale_bias_maxpool_cuda(
    torch::Tensor input, 
    torch::Tensor bias, 
    float scaling_factor, 
    int pool_size) {
    
    auto N = input.size(0);
    auto C = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    
    // Calculate output dimensions
    // Formula: floor((H_in - kernel_size) / stride + 1) assuming stride = kernel_size
    int H_out = (H_in - pool_size) / pool_size + 1;
    int W_out = (W_in - pool_size) / pool_size + 1;
    
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());
    
    int total_elements = N * C * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_tanh_scale_bias_maxpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out, scaling_factor, pool_size
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_tanh_scale_bias_maxpool_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor, int pool_size);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_tanh_scale_bias_maxpool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused post-convolution operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        # Standard Convolution
        x = self.conv(x)
        
        # Fused Tanh + Scale + Bias + MaxPool using custom CUDA kernel
        x = self.fused_ops.fused_tanh_scale_bias_maxpool_cuda(x, self.bias, self.scaling_factor, self.pool_kernel_size)
        
        return x