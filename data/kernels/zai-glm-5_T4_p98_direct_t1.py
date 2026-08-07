import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: Matmul -> AvgPool -> GELU -> Scale -> Max
# Optimized for speed using block-level matrix multiplication and parallel reductions.
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function for approximate GELU
__device__ __forceinline__ float gelu_approx(float x) {
    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Kernel for the fused operation
// Grid: (Batch, Blocks_per_Sample)
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
    
    // Map blockIdx.y to batch index, blockIdx.x to block index within the sample
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    int tid = threadIdx.x;
    int block_idx = blockIdx.x;
    int num_blocks_per_sample = gridDim.x;

    // Pointers for the current batch sample
    const float* input_row = input + batch_idx * in_features;
    
    // Shared memory for partial maxes.
    // We perform the max reduction across blocks using global memory (atomicMax or safe write),
    // but since we need to reduce 'pooled_out_features' (512) elements, and we have many blocks,
    // we can have each block compute a partial max and write to a global buffer, then reduce.
    // However, a simpler and effective strategy for 8192 in_features is to use enough blocks
    // to cover the output dimension efficiently.
    
    // Let's compute the Matmul output in a distributed way.
    // Each block computes a chunk of the output features.
    // Chunk size = ceil(out_features / num_blocks_per_sample)
    
    extern __shared__ float shared_mem[];
    // We use shared memory for two things:
    // 1. Partial dot products within the block (reduction)
    // 2. Storing the block's local max value
    
    float* s_data = shared_mem; // Size BLOCK_SIZE
    
    // Determine the range of output features this block handles
    int chunk_size = (out_features + num_blocks_per_sample - 1) / num_blocks_per_sample;
    int o_start = block_idx * chunk_size;
    int o_end = min(o_start + chunk_size, out_features);
    
    // We need to compute the max of the final processed values.
    // Since different blocks handle different parts of the output features,
    // they compute different parts of the pooled output.
    // We need to find the global max across all blocks for this sample.
    
    // Strategy:
    // 1. Each block computes its portion of the Matmul.
    // 2. Each block computes the corresponding portion of Pool -> GELU -> Scale.
    // 3. Each block finds the max within its portion.
    // 4. Each block writes its local max to a global buffer.
    // 5. A separate kernel or the last block reduces the buffer.
    // To avoid a second kernel launch, we can use atomicMax on the final output.
    
    float local_max = -INFINITY;

    // Iterate over the output features assigned to this block
    for (int o_idx = o_start; o_idx < o_end; ++o_idx) {
        // Compute dot product for matmul output o_idx
        // Parallel reduction over in_features
        float sum = 0.0f;
        
        // Strided loop over in_features
        for (int k = tid; k < in_features; k += blockDim.x) {
            sum += input_row[k] * weight[o_idx * in_features + k];
        }
        
        // Reduction within the block
        s_data[tid] = sum;
        __syncthreads();
        
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_data[tid] += s_data[tid + s];
            }
            __syncthreads();
        }
        
        // Thread 0 adds bias and finishes matmul
        float matmul_val = s_data[0];
        if (tid == 0) {
            matmul_val += bias[o_idx];
            
            // Now apply AvgPool, GELU, Scale
            // AvgPool: We need to know which pooled output index this belongs to.
            // This is tricky because AvgPool aggregates 'pool_kernel_size' inputs.
            // If we compute one matmul output at a time, we don't have the neighbors to average.
            
            // Correction: We must compute the full matmul output to perform AvgPool.
            // However, 'out_features' is 8192. We can't store it all in shared memory if we distribute blocks this way.
            
            // Alternative Strategy:
            // Use 1 block per sample? No, too slow for 8192x8192 matmul.
            // Use 1 block per pooled output element?
            // Pooled output size = 512.
            // If we launch 512 blocks per sample:
            // Each block computes one pooled output element.
            // To do that, it needs to compute 'pool_kernel_size' (16) matmul outputs.
            // It needs to compute 16 dot products of size 8192.
            // This is 16 * 8192 FMA. 
            // Block size 1024 threads -> each thread does 128 FMA. Very efficient.
            
            // Let's switch to this strategy.
            // Grid: (Batch, Pooled_Out_Features)
            // Block: (BLOCK_SIZE)
        }
    }
    
    // The logic above is broken for AvgPool. Let's restart the kernel logic with the correct strategy.
    // Correct Strategy: One block per pooled output element.
}

// Kernel: One block computes one element of the pooled output.
// Grid: (Batch, Pooled_Out_Features)
// Block: (BLOCK_SIZE)
__global__ void fused_op_kernel_v2(
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
    int pool_idx = blockIdx.x; // 0 to pooled_out_features - 1
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
        
        // Add bias
        if (tid == 0) {
            pool_sum += s_data[0] + bias[o_idx];
        }
        __syncthreads(); // Ensure all threads see updated s_data (though only tid 0 writes)
    }
    
    // Compute final value for this pooled element
    if (tid == 0) {
        float avg = pool_sum / pool_kernel_size;
        float g = gelu_approx(avg);
        float scaled = g * scale_factor;
        
        // We need to find the max across all pooled elements for this batch.
        // We can write to a temporary buffer and reduce, or use atomicMax directly.
        // Since we want the max of the whole output, we can use atomicMax.
        // However, atomicMax on float is not standard on all hardware or might be slow.
        // Better: Write to a global buffer, then reduce.
        
        // Use the output tensor as the buffer? 
        // Output is (batch_size). We need to reduce (batch_size, pooled_out_features) to (batch_size).
        // We can use atomicMax on a temporary buffer of size batch_size.
        // But we need to initialize it to -infinity.
        
        // Simplest fast approach: atomicMax.
        // Note: atomicMax operates on integers. We can use int float_as_int trick.
        // Or just write to global memory and launch a reduction kernel.
        // Given the constraints, let's write to a global scratchpad.
        // Actually, we can just write the result to output[batch_idx] using atomicMax.
        
        // Float atomicMax implementation
        // Since we are replacing the whole model, we can manage the output.
        // We initialize output to -infinity before launch.
        
        // atomicMax for float
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

// Kernel to initialize output buffer
__global__ void init_output_kernel(float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = -INFINITY;
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
    
    fused_op_kernel_v2<<<grid, block_size, shared_mem_size>>>(
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