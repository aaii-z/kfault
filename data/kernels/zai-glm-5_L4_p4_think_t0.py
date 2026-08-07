import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused double Mish
# Mish(x) = x * tanh(ln(1 + exp(x)))
# We fuse two Mish operations into a single kernel to save memory bandwidth.
double_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float mish_func(float x) {
    // log1pf(expf(x)) computes log(1 + exp(x)) accurately.
    // For large x, exp(x) -> inf, log(1+inf) -> inf, tanh(inf) -> 1. Result x * 1 = x.
    // For small x, standard precision.
    return x * tanhf(log1pf(expf(x)));
}

__global__ void double_mish_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Vectorized processing using float4 for better memory throughput
    int num_vec = size / 4;
    
    for (int i = idx; i < num_vec; i += stride) {
        // Load 4 floats
        float4 in_val = reinterpret_cast<const float4*>(input)[i];
        float4 out_val;
        
        // Apply double mish
        out_val.x = mish_func(mish_func(in_val.x));
        out_val.y = mish_func(mish_func(in_val.y));
        out_val.z = mish_func(mish_func(in_val.z));
        out_val.w = mish_func(mish_func(in_val.w));
        
        // Store 4 floats
        reinterpret_cast<float4*>(output)[i] = out_val;
    }
    
    // Handle remaining elements
    int rem_start = num_vec * 4;
    for (int i = rem_start + idx; i < size; i += stride) {
        output[i] = mish_func(mish_func(input[i]));
    }
}

torch::Tensor double_mish_cuda(torch::Tensor x) {
    // Ensure input is contiguous for the kernel logic to work correctly
    // Conv2d output is contiguous, but good practice to handle generally
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    auto out = torch::empty_like(x);
    int size = x.numel();
    
    const int block_size = 256;
    // Calculate grid size
    // We use grid-stride loop, so we can launch fewer blocks if needed, 
    // but covering the whole size with blocks is standard for simple kernels.
    int num_blocks = (size + block_size - 1) / block_size;
    
    double_mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
"""

double_mish_cpp_source = "torch::Tensor double_mish_cuda(torch::Tensor x);"

# Compile the inline CUDA code
double_mish_op = load_inline(
    name="double_mish",
    cpp_sources=double_mish_cpp_source,
    cuda_sources=double_mish_source,
    functions=["double_mish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.double_mish = double_mish_op

    def forward(self, x):
        x = self.conv(x)
        x = self.double_mish.double_mish_cuda(x)
        return x