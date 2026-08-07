import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the entire forward pass
fused_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Optimized Kernel Implementation
// Grid: batch_size blocks. Each block processes one sample.
// Block Dim: 256 threads.
__global__ void fused_forward_kernel_v2(
    const float* __restrict__ x_in, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    const float* __restrict__ y,
    float* __restrict__ out,
    int in_features, int out_features) {
    
    int b = blockIdx.x; // Batch index
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Pointers for this sample
    const float* x_ptr = x_in + b * in_features;
    const float* y_ptr = y + b * out_features;
    float* out_ptr = out + b * out_features;
    
    // Shared memory for partial sums (Linear) and reduction (IN)
    // We need to compute linear outputs first. 
    // Since out_features (8192) > block_size (256), each thread computes multiple output features.
    
    // 1. Linear Layer Computation
    // Each thread computes a subset of output features.
    // Loop over output features assigned to this thread.
    // We need a buffer to store the linear output before IN.
    // We can use global memory or large shared memory.
    // 8192 floats = 32KB shared memory. This fits in typical shared memory (48KB+).
    extern __shared__ float s_data[]; // Size: out_features
    
    // Compute Linear
    // Each thread loops over output features with stride block_size
    for (int f = tid; f < out_features; f += block_size) {
        const float* w_ptr = weight + f * in_features;
        float sum = bias[f];
        for (int k = 0; k < in_features; ++k) {
            sum += x_ptr[k] * w_ptr[k];
        }
        s_data[f] = sum; // Store linear output in shared memory
    }
    __syncthreads();
    
    // 2. Instance Normalization
    // Compute mean and var of s_data[0...out_features]
    // Parallel reduction over s_data
    
    float sum_local = 0.0f;
    for (int f = tid; f < out_features; f += block_size) {
        sum_local += s_data[f];
    }
    
    // Reduction in shared memory
    __shared__ float s_sum[256]; // Max block size
    s_sum[tid] = sum_local;
    __syncthreads();
    
    for (int s = block_size/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    
    float mean = s_sum[0] / out_features;
    
    // Compute variance
    float var_sum_local = 0.0f;
    for (int f = tid; f < out_features; f += block_size) {
        float diff = s_data[f] - mean;
        var_sum_local += diff * diff;
    }
    
    __shared__ float s_var_sum[256];
    s_var_sum[tid] = var_sum_local;
    __syncthreads();
    
    for (int s = block_size/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_var_sum[tid] += s_var_sum[tid + s];
        }
        __syncthreads();
    }
    
    float var = s_var_sum[0] / out_features;
    float inv_std = rsqrtf(var + 1e-5f);
    
    // 3. Normalize, Add, Mul
    for (int f = tid; f < out_features; f += block_size) {
        float norm_val = (s_data[f] - mean) * inv_std;
        float y_val = y_ptr[f];
        out_ptr[f] = (norm_val + y_val) * y_val;
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor y) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    
    auto out = torch::empty({batch_size, out_features}, x.options());
    
    const int block_size = 256;
    int shared_mem_size = out_features * sizeof(float); // For storing linear outputs
    
    fused_forward_kernel_v2<<<batch_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        in_features,
        out_features
    );
    
    return out;
}
"""

fused_forward_cpp_source = (
    "torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor y);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_forward_cpp_source,
    cuda_sources=fused_forward_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Keep the Linear layer to maintain the same weight initialization as the original model
        self.bmm = nn.Linear(in_features, out_features)
        self.fused_ops = fused_ops

    def forward(self, x, y):
        # Extract weight and bias from the Linear layer to pass to the CUDA kernel
        return self.fused_ops.fused_forward_cuda(x, self.bmm.weight, self.bmm.bias, y)