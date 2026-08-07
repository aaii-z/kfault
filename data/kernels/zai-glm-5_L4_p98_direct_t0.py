import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation: Matmul -> AvgPool -> GELU -> Scale -> Max
# This kernel fuses the entire forward pass into a single kernel launch.
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function for approximate GELU (faster than exact GELU)
__device__ __forceinline__ float gelu_approx(float x) {
    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Kernel for the fused operation
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
    
    // Each block handles one sample in the batch
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Shared memory for the intermediate result after matmul, pool, gelu, scale
    // We need to store the 'out_features' dimension to perform the final max reduction
    extern __shared__ float shared_data[];

    // Thread index within the block
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    // Pointers for the current batch sample
    const float* input_row = input + batch_idx * in_features;
    
    // Step 1: Matmul (Linear)
    // Each thread computes a portion of the output features
    for (int i = tid; i < out_features; i += block_dim) {
        float sum = 0.0f;
        const float* weight_col = weight + i * in_features;
        
        // Dot product of input row and weight column
        for (int k = 0; k < in_features; ++k) {
            sum += input_row[k] * weight_col[k];
        }
        
        // Add bias
        sum += bias[i];
        
        // Step 2: AvgPool1d
        // The AvgPool is applied on the dimension of size out_features.
        // The input to AvgPool is conceptually (1, out_features).
        // Kernel size is pool_kernel_size.
        // We are computing the output at index 'i'.
        // The output index 'i' corresponds to the pooled result of the input window.
        // Input window for output 'i' starts at 'i * pool_kernel_size'.
        // However, the AvgPool is applied to the result of the matmul.
        // Wait, the AvgPool is applied to the matmul output.
        // The matmul output size is out_features.
        // The AvgPool reduces this dimension.
        // Let's re-read the PyTorch logic:
        // x = self.matmul(x) -> (batch, out_features)
        // x = self.avg_pool(x.unsqueeze(1)).squeeze(1) -> AvgPool1d on the last dim.
        // Input to pool: (batch, 1, out_features). Kernel size = pool_kernel_size.
        // Output of pool: (batch, 1, out_features / pool_kernel_size). Squeezed: (batch, out_features / pool_kernel_size).
        // GELU and Scale apply element-wise.
        // Max(dim=1) reduces the last dimension.
        // Final output shape: (batch, 1) -> squeezed to (batch,).
        
        // Correction: The logic above implies the loop should iterate over the *pooled* output dimension.
        // Let 'pooled_out_features' = out_features / pool_kernel_size.
        // The kernel needs to compute the matmul, then pool, then gelu, then scale, then max.
        
        // Let's adjust the indexing.
        // We need to compute the matmul for the full 'out_features' first? No, we can fuse.
        // For a specific output index 'j' of the pooled output (0 to pooled_out_features - 1):
        // It depends on matmul output indices from 'j * pool_kernel_size' to '(j+1) * pool_kernel_size - 1'.
        
        // Optimization:
        // Each thread will compute one element of the 'pooled_out_features' dimension.
        // To do this, the thread needs to compute 'pool_kernel_size' dot products (the matmul outputs) and average them.
        
        // Let's map threads to the pooled output dimension.
        // If tid < pooled_out_features:
        //   Compute average of matmul outputs in the range [tid * pool_kernel_size, (tid+1) * pool_kernel_size).
        
        // BUT, 'out_features' is 8192 and 'pool_kernel_size' is 16. 'pooled_out_features' is 512.
        // We have many threads (e.g., 256 or 512). We can parallelize differently.
        // Better approach:
        // 1. Each thread computes a partial sum for the matmul.
        // 2. Reduction within the block to get the full matmul output.
        // 3. Then apply pool, gelu, scale.
        // 4. Then max reduction.
        
        // Actually, given the sizes (Batch=1024, In=8192, Out=8192), the matmul is the most expensive part.
        // The output of the whole network is (batch_size).
        // We want to compute one float per batch item.
        
        // Strategy:
        // Use one block per batch item.
        // Threads in the block cooperate to compute the matmul output.
        // Then apply the subsequent layers.
        
        // Dimensions:
        // Matmul output size: 8192 floats.
        // Pooled output size: 512 floats.
        // We need to compute the matmul output to proceed.
        
        // Shared memory usage:
        // We can store the matmul output in shared memory? 8192 * 4 bytes = 32KB. This fits in shared memory (usually 48KB+).
        // So, plan:
        // 1. Each thread computes a portion of the matmul output (8192 floats total).
        //    Store results in shared memory 'shared_matmul_out'.
        // 2. __syncthreads().
        // 3. Each thread takes a portion of 'shared_matmul_out', applies AvgPool, GELU, Scale.
        //    Store results in a second shared memory buffer 'shared_processed'.
        //    Size of 'shared_processed': 512 * 4 bytes = 2KB.
        // 4. __syncthreads().
        // 5. Parallel reduction (Max) on 'shared_processed' to get the final output for the batch item.
        
        // Shared memory allocation:
        // We need 8192 floats for matmul output.
        // We need 512 floats for processed output.
        // Total shared memory per block: (8192 + 512) * 4 bytes = 34,816 bytes ~ 35KB. Safe.
        
        // Renaming shared_data to be explicit about layout.
        // We'll use the extern shared array. Layout: [matmul_out (8192), processed_out (512)].
        
        float* s_matmul = shared_data;
        float* s_processed = shared_data + out_features;
        
        // Step 1: Compute Matmul
        // Distribute 'out_features' (8192) across threads.
        // Each thread computes multiple elements if necessary.
        for (int o_idx = tid; o_idx < out_features; o_idx += block_dim) {
            float sum = 0.0f;
            // Dot product input (8192) and weight column (8192)
            // Loop over in_features
            // Optimization: Use vectorized loads? For simplicity, standard loop first.
            // To improve speed, we should probably use a more optimized matmul, but let's stick to the logic first.
            for (int k = 0; k < in_features; ++k) {
                sum += input_row[k] * weight[o_idx * in_features + k];
            }
            sum += bias[o_idx];
            s_matmul[o_idx] = sum;
        }
        
        __syncthreads();
        
        // Step 2, 3, 4: AvgPool -> GELU -> Scale
        // Distribute 'pooled_out_features' (512) across threads.
        int pooled_out_features = out_features / pool_kernel_size;
        
        // Initialize processed values to a very small number for max reduction later
        // (though we will compute valid values for all indices)
        float local_max = -INFINITY;
        
        // Each thread handles one or more pooled output indices
        for (int p_idx = tid; p_idx < pooled_out_features; p_idx += block_dim) {
            // AvgPool
            float pool_sum = 0.0f;
            int start_idx = p_idx * pool_kernel_size;
            for (int k = 0; k < pool_kernel_size; ++k) {
                pool_sum += s_matmul[start_idx + k];
            }
            float avg = pool_sum / pool_kernel_size;
            
            // GELU
            float g = gelu_approx(avg);
            
            // Scale
            float scaled = g * scale_factor;
            
            s_processed[p_idx] = scaled;
        }
        
        __syncthreads();
        
        // Step 5: Max reduction
        // Reduce 's_processed' (size 512) to a single float.
        // Parallel reduction in shared memory.
        
        // We need to find the max of s_processed[0...pooled_out_features-1].
        // We can perform parallel reduction.
        // Threads load values into registers first.
        
        // Ensure initialization for threads that might not enter loops
        float max_val = -INFINITY;
        
        // Simple reduction loop
        // Each thread reduces a portion, then we reduce the partial results.
        // Or just standard parallel reduction.
        
        // Since pooled_out_features is small (512), we can do a simple loop or reduction.
        // If block_dim >= pooled_out_features, we can do a single reduction step.
        // block_dim is likely 256 or 512. pooled_out_features is 512.
        
        // Approach: Strided reduction in shared memory.
        // We need to handle cases where block_dim < pooled_out_features.
        
        // First, load values to max_val
        for (int i = tid; i < pooled_out_features; i += block_dim) {
            max_val = fmaxf(max_val, s_processed[i]);
        }
        
        // Now reduce 'max_val' across threads in the block.
        // We can use shared memory for this reduction.
        // Re-use s_matmul (we don't need it anymore) or part of s_processed?
        // Let's use the beginning of shared_data for partial maxes.
        float* s_partial_maxes = shared_data; 
        
        s_partial_maxes[tid] = max_val;
        __syncthreads();
        
        // Reduce
        // Assuming block_dim is power of 2 (e.g., 256, 512).
        for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_partial_maxes[tid] = fmaxf(s_partial_maxes[tid], s_partial_maxes[tid + s]);
            }
            __syncthreads();
        }
        
        // Write result
        if (tid == 0) {
            output[batch_idx] = s_partial_maxes[0];
        }
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
    int out_features = weight.size(0); // weight is (out_features, in_features)
    
    auto output = torch::empty({batch_size}, input.options());
    
    // Shared memory size: matmul output + processed output (for reduction)
    // Actually, the kernel uses shared_data for:
    // 1. Matmul output: out_features * sizeof(float)
    // 2. Processed output: (out_features / pool_kernel_size) * sizeof(float)
    // 3. Partial maxes: block_dim * sizeof(float) (reuses space)
    // Total size needed = out_features * 4 + (out_features / pool_kernel_size) * 4.
    // We don't need to store partial maxes separately if we reuse the matmul buffer.
    // Max size = out_features + pooled_out_features.
    
    int pooled_out_features = out_features / pool_kernel_size;
    size_t shared_mem_size = (out_features + pooled_out_features) * sizeof(float);
    
    const int block_size = 256; // 256 threads per block
    const int num_blocks = batch_size;
    
    fused_op_kernel<<<num_blocks, block_size, shared_mem_size>>>(
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
        # We keep the parameters as buffers or parameters to maintain the interface,
        # but we will use them in the custom CUDA kernel.
        # Note: nn.Linear creates weight (out, in) and bias (out).
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        # Initialize parameters similar to nn.Linear (Kaiming uniform)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(x, self.weight, self.bias, self.pool_kernel_size, self.scale_factor)

import math # Ensure math is imported for init