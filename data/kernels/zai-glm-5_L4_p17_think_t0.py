import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Instance Normalization and Division
fused_instance_norm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel for fused Instance Normalization and Division
// Normalizes per channel per sample and divides by a constant
__global__ void instance_norm_div_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                         int N, int C, int H, int W, float divide_by, float epsilon) {
    // Each block handles one (n, c) channel
    int n = blockIdx.x;
    int c = blockIdx.y;
    
    int channel_size = H * W;
    int offset = (n * C + c) * channel_size;
    
    const float* x = input + offset;
    float* y = output + offset;
    
    // 1. Calculate Mean and Variance
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Strided loop to handle large channel sizes with limited threads
    for (int i = threadIdx.x; i < channel_size; i += blockDim.x) {
        float val = x[i];
        sum += val;
        sum_sq += val * val;
    }
    
    // Block reduction for sum and sum_sq
    // Using shared memory for reduction
    __shared__ float s_sum[256]; // Assuming block size <= 256
    __shared__ float s_sum_sq[256];
    
    s_sum[threadIdx.x] = sum;
    s_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();
    
    // Reduction (tree based)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 computes mean and variance
    __shared__ float mean;
    __shared__ float inv_std_div; // Combined factor: 1 / (std * divide_by)
    
    if (threadIdx.x == 0) {
        mean = s_sum[0] / channel_size;
        float var = s_sum_sq[0] / channel_size - mean * mean;
        // Using rsqrtf for 1/sqrt(x) optimization
        // Factor = 1 / (sqrt(var + eps) * divide_by)
        inv_std_div = rsqrtf(var + epsilon) / divide_by;
    }
    __syncthreads();
    
    // 2. Normalize and Divide
    // y = (x - mean) * inv_std_div
    // y = x * inv_std_div - mean * inv_std_div
    float scale = inv_std_div;
    float bias = -mean * inv_std_div;
    
    for (int i = threadIdx.x; i < channel_size; i += blockDim.x) {
        y[i] = x[i] * scale + bias;
    }
}

// C++ wrapper function
torch::Tensor instance_norm_div_cuda(torch::Tensor input, float divide_by) {
    // Ensure input is contiguous and on CUDA
    auto input_contiguous = input.contiguous();
    auto output = torch::empty_like(input_contiguous);
    
    int N = input_contiguous.size(0);
    int C = input_contiguous.size(1);
    int H = input_contiguous.size(2);
    int W = input_contiguous.size(3);
    
    float epsilon = 1e-5; // Standard epsilon for Instance Norm
    
    // Grid dimensions: (N, C)
    dim3 grid(N, C);
    // Block dimension: 256 threads per block
    dim3 block(256);
    
    instance_norm_div_kernel<<<grid, block>>>(
        input_contiguous.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N, C, H, W, 
        divide_by, epsilon
    );
    
    return output;
}
"""

fused_instance_norm_div_cpp_source = (
    "torch::Tensor instance_norm_div_cuda(torch::Tensor input, float divide_by);"
)

# Compile the inline CUDA code
fused_instance_norm_div = load_inline(
    name="fused_instance_norm_div",
    cpp_sources=fused_instance_norm_div_cpp_source,
    cuda_sources=fused_instance_norm_div_source,
    functions=["instance_norm_div_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused Instance Normalization and Division.
    The Conv2d operation is kept as the standard PyTorch implementation (cuDNN) as it is 
    highly optimized and fusing it with the subsequent normalization is complex and 
    often does not yield significant benefits over the memory bandwidth savings gained 
    by fusing the post-processing steps.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.fused_op = fused_instance_norm_div

    def forward(self, x):
        x = self.conv(x)
        # Replace InstanceNorm and Division with the fused CUDA kernel
        x = self.fused_op.instance_norm_div_cuda(x, self.divide_by)
        return x