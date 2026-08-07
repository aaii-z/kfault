import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Hardtanh, Mean, and Tanh
# This kernel fuses the element-wise operations (Hardtanh, Tanh) with the reduction (Mean)
# to minimize memory bandwidth usage and kernel launch overhead.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: Hardtanh -> Mean(dim 2,3) -> Tanh
// Assumes input tensor is 4D (N, C, H, W) and output is 4D (N, C, 1, 1)
__global__ void fused_hardtanh_mean_tanh_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int N, int C, int H, int W, 
    float hardtanh_min, float hardtanh_max) {
    
    // Map each thread block to a specific (n, c) channel
    // blockIdx.x ranges from 0 to N*C - 1
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc % C;
    
    // Pointers to the start of the input channel and output element
    const float* in_ptr = input + nc * H * W;
    float* out_ptr = output + nc;
    
    // Use shared memory for reduction within the block
    extern __shared__ float sdata[];
    
    // Stride loop to perform Hardtanh and partial sum
    // Each thread processes multiple elements if H*W > block_size
    float thread_sum = 0.0f;
    int total_elements = H * W;
    
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        float val = in_ptr[i];
        // Hardtanh clamping
        val = fmaxf(hardtanh_min, fminf(hardtanh_max, val));
        thread_sum += val;
    }
    
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Block reduction to sum the partial results
    // Assuming blockDim.x is a power of 2
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 computes the mean and applies tanh
    if (threadIdx.x == 0) {
        float mean_val = sdata[0] / total_elements;
        // Tanh activation
        *out_ptr = tanhf(mean_val);
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, 
    float hardtanh_min, 
    float hardtanh_max) {
    
    // Input dimensions
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    // Output tensor shape (N, C, 1, 1)
    auto output = torch::empty({N, C, 1, 1}, input.options());
    
    // Launch configuration
    // One block per (N, C) channel
    int blocks = N * C;
    
    // Threads per block. 256 is a good default for occupancy.
    int threads = 256; 
    
    // Shared memory size for reduction
    int shared_mem_size = threads * sizeof(float);
    
    // Launch kernel
    fused_hardtanh_mean_tanh_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        hardtanh_min, hardtanh_max
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, float hardtanh_min, float hardtanh_max);"
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
    Optimized Model that performs a transposed convolution, followed by max pooling,
    and a fused Hardtanh-Mean-Tanh operation implemented with a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        
        # Store hardtanh limits for the custom kernel
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Load the custom operator
        self.fused_ops = fused_ops

    def forward(self, x):
        # Standard PyTorch layers for ConvTranspose and MaxPool
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        
        # Custom CUDA kernel for fused Hardtanh, Mean, and Tanh
        x = self.fused_ops.fused_ops_cuda(x, self.hardtanh_min, self.hardtanh_max)
        
        return x