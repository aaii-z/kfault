import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the entire forward pass
fused_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel for fused operation: Linear -> Instance Norm -> Add -> Mul
// Optimized for large batch_size (1024) and features (8192)
__global__ void fused_forward_kernel(
    const float* __restrict__ x_in, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    const float* __restrict__ y,
    float* __restrict__ out,
    int batch_size, int in_features, int out_features) {
    
    // 1. Compute Linear Layer (BMM)
    // Each block handles one output feature for a batch of samples? 
    // Or each block handles one sample's output features?
    // Given batch=1024, out=8192. Total output elements = 8,388,608.
    // Let's map one thread to one output element.
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_features;
    
    if (idx < total_outputs) {
        int b = idx / out_features; // Batch index
        int f = idx % out_features; // Feature index
        
        // Compute Linear: out_linear = x_in[b, :] @ weight[:, f] + bias[f]
        // x_in shape: (batch_size, in_features)
        // weight shape: (out_features, in_features) - PyTorch Linear stores weight as (out, in)
        
        const float* x_ptr = x_in + b * in_features;
        const float* w_ptr = weight + f * in_features;
        
        float linear_val = 0.0f;
        
        // Vectorized load for coalesced memory access if possible, 
        // but here x_ptr and w_ptr are strided differently for each thread in a warp.
        // Simple loop is standard for GEMM parts if not using shared memory tiling.
        // Given the complexity of writing a full GEMM kernel with tiling in a single fused kernel,
        // we will rely on the high memory bandwidth and simple loop for now, 
        // or assume the compiler might optimize.
        // However, for 8192 features, this loop is long.
        
        // Optimization: Loop unrolling or vectorization could be added, 
        // but keeping it simple for correctness and compilation stability.
        for (int k = 0; k < in_features; ++k) {
            linear_val += x_ptr[k] * w_ptr[k];
        }
        linear_val += bias[f];
        
        // 2. Instance Normalization
        // We need mean and variance across the feature dimension for the current batch item 'b'.
        // This requires synchronization across threads processing the same 'b'.
        // Threads with same 'b' are contiguous in 'idx' (idx = b*out_features + f).
        // Block size is 256. out_features is 8192.
        // Threads for the same 'b' span multiple blocks.
        // Standard Instance Norm requires global sync or two-pass approach.
        // Since we are fusing, we can't easily do two-pass without writing intermediate results.
        // However, we can use a parallel reduction approach within blocks, 
        // but since threads for one 'b' span multiple blocks, we'd need atomic operations 
        // or global memory writes to compute mean/var.
        
        // Given the constraints, let's use a simplified approach or atomic operations 
        // if we were to compute it. But computing mean/var globally for each sample 
        // inside a kernel that also does GEMM is complex.
        // 
        // Alternative: The problem asks to replace operators. 
        // We can fuse the Add and Mul easily. 
        // Fusing Instance Norm is hard because of the dependency on the output of Linear 
        // for the whole sample.
        // 
        // Let's try to compute mean and var using atomic adds to global memory.
        // This is slow but correct for a fused kernel without multiple kernel launches.
        // However, atomics on FP32 are supported.
        
        // Wait, the prompt implies we can replace "multiple operators".
        // If we can't efficiently fuse Instance Norm due to global reduction, 
        // we might just fuse Linear + Add + Mul? 
        // But the architecture is Linear -> IN -> Add -> Mul.
        // The IN output is input to Add.
        // We must compute IN.
        
        // Let's use a block-level reduction for mean/var. 
        // We need to launch kernels differently for that.
        // If we launch (batch_size) blocks, each block computes one sample.
        // Block size = 1024? No, max threads is 1024.
        // Features = 8192. We need reduction over 8192 elements.
        // We can use shared memory reduction.
        
        // RE-THINKING KERNEL GRID:
        // Grid: (batch_size). Block: (256) or (512) or (1024).
        // Each block handles one sample.
        // Threads in block cooperate to compute Linear, then reduce for IN, then Add/Mul.
        
        // This requires rewriting the kernel logic entirely.
        // Let's implement this optimized grid strategy.
    }
}

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
        out_ptr[f] = norm_val * y_val + y_val; // (x * y) + y ? No, x = x + y; x = x * y.
        // Original: x = x + y; x = x * y;
        // x is norm_val.
        // x = norm_val + y_val
        // x = x * y_val = (norm_val + y_val) * y_val
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
        # We keep the parameters but will use our custom kernel
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_ops = fused_ops

    def forward(self, x, y):
        return self.fused_ops.fused_forward_cuda(x, self.weight, self.bias, y)