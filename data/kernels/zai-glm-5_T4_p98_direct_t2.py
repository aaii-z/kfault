import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: Matmul -> AvgPool -> GELU -> Scale -> Max
# Corrected logic: AvgPool is performed on the 'features' dimension (dim 1), not the batch dimension.
# Input: (Batch, In_Features)
# Matmul: (Batch, Out_Features)
# AvgPool1d: (Batch, 1, Out_Features) -> (Batch, 1, Out_Features / Pool_Kernel) -> (Batch, Pooled_Features)
# GELU, Scale: Element-wise
# Max: (Batch, Pooled_Features) -> (Batch, 1) -> (Batch)

fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function for approximate GELU
__device__ __forceinline__ float gelu_approx(float x) {
    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Kernel to initialize output buffer to -infinity
__global__ void init_output_kernel(float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = -INFINITY;
    }
}

// Kernel for the fused operation
// Grid: (Batch, Pooled_Features)
// Block: (BLOCK_SIZE)
__global__ void fused_op_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, 
    int in_features, 
    int out_features, 
    int pool_kernel_size, 
    float scale_factor) {
    
    int batch_idx = blockIdx.y;
    int pool_idx = blockIdx.x; // Index in the pooled features dimension
    int pooled_out_features = gridDim.x;
    
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    
    // Shared memory for reduction
    extern __shared__ float s_data[];
    
    const float* input_row = input + batch_idx * in_features;
    
    // We need to compute the average of pool_kernel_size matmul outputs.
    // Matmul output indices: [pool_idx * pool_kernel_size, (pool_idx + 1) * pool_kernel_size)
    
    float pool_sum = 0.0f;
    
    // Loop over the pool window
    for (int p = 0; p < pool_kernel_size; ++p) {
        int o_idx = pool_idx * pool_kernel_size + p;
        
        // Compute matmul output for o_idx
        // Dot product of input_row (in_features) and weight[o_idx]
        float sum = 0.0f;
        
        // Parallel reduction over in_features
        for (int k = tid; k < in_features; k += blockDim.x) {
            sum += input_row[k] * weight[o_idx * in_features + k];
        }
        
        s_data[tid] = sum;
        __syncthreads();
        
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_data[tid] += s_data[tid + s];
            }
            __syncthreads();
        }
        
        // Add bias and accumulate for pool
        if (tid == 0) {
            pool_sum += s_data[0] + bias[o_idx];
        }
        __syncthreads(); // Ensure all threads are ready for next iteration
    }
    
    // Compute final value for this pooled element
    if (tid == 0) {
        float avg = pool_sum / pool_kernel_size;
        float g = gelu_approx(avg);
        float scaled = g * scale_factor;
        
        // Reduce max across pooled features for this batch sample
        // Use atomicMax on the output buffer (initialized to -inf)
        float* addr = output + batch_idx;
        float old = *addr;
        float assumed;
        do {
            assumed = old;
            if (scaled <= assumed) break;
            old = atomicCAS((int*)addr, __float_as_int(assumed), __float_as_int(scaled));
        } while (old != assumed);
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int pool_kernel_size, 
    float scale_factor) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size}, input.options());
    
    // Initialize output to -infinity for atomicMax
    const int init_block_size = 256;
    int init_num_blocks = (batch_size + init_block_size - 1) / init_block_size;
    init_output_kernel<<<init_num_blocks, init_block_size>>>(output.data_ptr<float>(), batch_size);
    
    int pooled_out_features = out_features / pool_kernel_size;
    
    const int block_size = 256; // Threads per block
    size_t shared_mem_size = block_size * sizeof(float);
    
    dim3 grid(pooled_out_features, batch_size);
    
    fused_op_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        pool_kernel_size,
        scale_factor
    );
    
    return output;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int pool_kernel_size, float scale_factor);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        # Initialize parameters similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(x, self.weight, self.bias, self.pool_kernel_size, self.scale_factor)