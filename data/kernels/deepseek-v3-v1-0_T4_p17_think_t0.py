import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom fused InstanceNorm + Scale kernel
instance_norm_scale_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

template<typename T>
__global__ void instance_norm_scale_kernel(
    const T* input,
    T* output,
    const int num_instances,
    const int instance_size,
    const float eps,
    const float scale_factor) {
    
    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = &shared[blockDim.x];
    
    const int instance_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Compute mean
    float mean = 0.0f;
    for (int i = tid; i < instance_size; i += blockDim.x) {
        mean += static_cast<float>(input[instance_idx * instance_size + i]);
    }
    
    mean_shared[tid] = mean;
    __syncthreads();
    
    // Reduce mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mean_shared[tid] += mean_shared[tid + s];
        }
        __syncthreads();
    }
    
    mean = mean_shared[0] / instance_size;
    __syncthreads();
    
    // Compute variance
    float variance = 0.0f;
    for (int i = tid; i < instance_size; i += blockDim.x) {
        float diff = static_cast<float>(input[instance_idx * instance_size + i]) - mean;
        variance += diff * diff;
    }
    
    var_shared[tid] = variance;
    __syncthreads();
    
    // Reduce variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            var_shared[tid] += var_shared[tid + s];
        }
        __syncthreads();
    }
    
    variance = var_shared[0] / instance_size;
    float inv_std = rsqrtf(variance + eps);
    
    // Normalize and scale
    for (int i = tid; i < instance_size; i += blockDim.x) {
        output[instance_idx * instance_size + i] = 
            (static_cast<float>(input[instance_idx * instance_size + i]) - mean) * 
            inv_std * scale_factor;
    }
}

torch::Tensor instance_norm_scale_cuda(
    torch::Tensor input,
    float eps,
    float scale_factor) {
    
    auto output = torch::empty_like(input);
    const int num_instances = input.size(0) * input.size(1);
    const int instance_size = input.size(2) * input.size(3);
    
    dim3 grid(num_instances);
    dim3 block(BLOCK_SIZE);
    size_t shared_mem = 2 * BLOCK_SIZE * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "instance_norm_scale", [&] {
        instance_norm_scale_kernel<scalar_t><<<grid, block, shared_mem>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            num_instances,
            instance_size,
            eps,
            scale_factor);
    });
    
    return output;
}
"""

instance_norm_scale_cpp_source = """
torch::Tensor instance_norm_scale_cuda(torch::Tensor input, float eps, float scale_factor);
"""

# Load the custom CUDA extension
instance_norm_scale = load_inline(
    name="instance_norm_scale",
    cpp_sources=instance_norm_scale_cpp_source,
    cuda_sources=instance_norm_scale_source,
    functions=["instance_norm_scale_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.eps = 1e-5
        self.instance_norm_scale = instance_norm_scale

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm_scale.instance_norm_scale_cuda(x, self.eps, 1.0/self.divide_by)
        return x