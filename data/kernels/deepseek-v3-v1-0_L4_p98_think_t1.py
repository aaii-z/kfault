import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for GELU+Scale+Max operations
gelu_scale_max_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__global__ void gelu_scale_max_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_windows,
    float scale_factor
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid >= batch_size) return;
    
    extern __shared__ float shared[];
    float* shared_vals = shared;
    
    // Initialize shared memory
    shared_vals[tid] = -FLT_MAX;
    __syncthreads();
    
    // Each thread processes multiple elements
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_windows; i += blockDim.x) {
        float val = input[bid * num_windows + i];
        float gelu_val = gelu(val);
        float scaled_val = gelu_val * scale_factor;
        if (scaled_val > local_max) local_max = scaled_val;
    }
    
    // Reduce within warp
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }
    
    // First thread in warp writes to shared memory
    if (tid % 32 == 0) {
        shared_vals[tid / 32] = local_max;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < 32) {
        float warp_max = shared_vals[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, warp_max, offset);
            if (other > warp_max) warp_max = other;
        }
        if (tid == 0) {
            output[bid] = warp_max;
        }
    }
}

torch::Tensor gelu_scale_max_cuda(
    torch::Tensor input,
    float scale_factor
) {
    int batch_size = input.size(0);
    int num_windows = input.size(1);
    auto output = torch::empty({batch_size}, input.options());
    
    dim3 blocks(batch_size);
    dim3 threads(256);
    int shared_mem_size = (threads.x / 32) * sizeof(float);
    
    gelu_scale_max_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_windows,
        scale_factor
    );
    
    return output;
}
"""

gelu_scale_max_cpp = """
torch::Tensor gelu_scale_max_cuda(torch::Tensor input, float scale_factor);
"""

# Load the custom CUDA operation
gelu_scale_max_ops = load_inline(
    name='gelu_scale_max_ops',
    cpp_sources=gelu_scale_max_cpp,
    cuda_sources=gelu_scale_max_source,
    functions=['gelu_scale_max_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        assert out_features % pool_kernel_size == 0, "out_features must be divisible by pool_kernel_size"
        
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.num_windows = out_features // pool_kernel_size
        
        # Initialize linear layer parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Compute averaged weight and bias
        weight_avg = self.weight.view(self.num_windows, self.pool_kernel_size, self.in_features).mean(dim=1)
        bias_avg = self.bias.view(self.num_windows, self.pool_kernel_size).mean(dim=1)
        
        # Apply linear transformation
        x = torch.nn.functional.linear(x, weight_avg, bias_avg)
        
        # Apply fused GELU+Scale+Max
        return gelu_scale_max_ops.gelu_scale_max_cuda(x, self.scale_factor)