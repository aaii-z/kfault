import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: Sub, HardSwish, MaxPool, Mish
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float hardswish_device(float x) {
    // x * relu6(x + 3) / 6
    float val = x + 3.0f;
    val = fmaxf(0.0f, val);
    val = fminf(6.0f, val);
    return x * val / 6.0f;
}

__device__ float mish_device(float x) {
    // x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    // Using log1p(exp(x)) for numerical stability
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_sub_hardswish_maxpool_mish_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    const float subtract_value,
    const int pool_size,
    const int B, const int C, 
    const int H_in, const int W_in,
    const int H_out, const int W_out) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * H_out * W_out;
    
    if (idx < total_elements) {
        // Decode linear index to 4D coordinates (n, c, h_out, w_out)
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int b = idx / (W_out * H_out * C);
        
        int h_in_start = h_out * pool_size;
        int w_in_start = w_out * pool_size;
        
        float max_val = -INFINITY;
        
        // Iterate over the pooling window
        for (int ph = 0; ph < pool_size; ++ph) {
            for (int pw = 0; pw < pool_size; ++pw) {
                int h_in = h_in_start + ph;
                int w_in = w_in_start + pw;
                
                // Calculate input index assuming contiguous NCHW memory layout
                long long in_idx = (long long)b * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                
                float val = input[in_idx];
                val -= subtract_value;
                val = hardswish_device(val);
                
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        
        float res = mish_device(max_val);
        output[idx] = res;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, float subtract_value, int pool_size) {
    // Input dimensions
    int B = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    // Output dimensions
    int H_out = H_in / pool_size;
    int W_out = W_in / pool_size;
    
    auto output = torch::zeros({B, C, H_out, W_out}, input.options());
    
    int total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_sub_hardswish_maxpool_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        subtract_value,
        pool_size,
        B, C, H_in, W_in, H_out, W_out
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, float subtract_value, int pool_size);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, followed by a fused kernel
    performing subtraction, HardSwish, MaxPool, and Mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # Apply fused CUDA kernel for Sub, HardSwish, MaxPool, Mish
        x = self.fused_ops.fused_op_cuda(x, self.subtract_value, self.pool_kernel_size)
        return x