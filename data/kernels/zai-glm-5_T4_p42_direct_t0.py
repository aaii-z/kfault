import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the sequence of operations
custom_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel for Global Average Pooling + Bias + LogSumExp + Sum + Mul
// Fusing these operations avoids writing intermediate results to global memory
// and minimizes kernel launch overhead.
__global__ void fused_ops_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, 
    int out_channels, 
    int height, 
    int width) {
    
    // Each block processes one sample in the batch
    int n = blockIdx.x;
    // Each thread processes a subset of channels for the reduction
    int c_start = threadIdx.x;
    int stride = blockDim.x;

    // Pointers for the current sample
    const float* input_n = input + n * out_channels * height * width;
    const float* bias_ptr = bias; // Bias is (C, 1, 1)

    // Shared memory for block-level reduction
    // We need to reduce over channels (dim 1) after the spatial reduction
    extern __shared__ float s_data[];

    // Initialize shared memory for this thread's partial reduction
    // We use s_data to store the result of logsumexp for each channel handled by a thread
    // However, we need to reduce over channels eventually.
    // Strategy:
    // 1. Each thread calculates spatial mean for its assigned channels.
    // 2. Each thread adds bias and computes logsumexp (which is just log(exp(x)) = x if we reduce over channels? No, wait).
    // Let's re-read the logic.
    // Original:
    // x = mean(x, (H, W)) -> (N, C, 1, 1)
    // x = x + bias -> (N, C, 1, 1)
    // x = logsumexp(x, 1) -> (N, 1, 1, 1)  <-- This reduces over channels
    // x = sum(x, (2, 3)) -> (N, 1)          <-- Sum over 1x1 spatial dims, effectively identity
    // x = x * 10.0
    
    // Correct Logic:
    // 1. Calculate spatial mean for each channel.
    // 2. Add bias.
    // 3. Reduce over channels using LogSumExp.
    // 4. The result is a single scalar per batch item (N).
    // 5. Multiply by 10.0.

    // We need a two-stage reduction or a parallel reduction in shared memory.
    // Stage 1: Threads compute spatial mean + bias for their channels.
    // Stage 2: Threads perform LogSumExp reduction over channels.
    
    // To optimize LogSumExp, we use the max trick.
    // LSE = log(Sum(exp(x))) = max + log(Sum(exp(x - max)))
    
    // We need shared memory to handle the channel reduction.
    // Let's use s_data to store partial results.
    // We need to find the max value across channels first, then sum exp(x - max).
    // This requires synchronization.
    
    // Shared memory layout:
    // [0 ... blockDim.x - 1]: max values
    // [blockDim.x ... 2*blockDim.x - 1]: sum exp values
    
    float local_max = -FLT_MAX;
    float local_sum_exp = 0.0f;

    // Loop over channels assigned to this thread
    for (int c = c_start; c < out_channels; c += stride) {
        // 1. Spatial Mean
        float spatial_sum = 0.0f;
        const float* channel_ptr = input_n + c * height * width;
        
        // Vectorizable loop over spatial dimensions
        // Unrolling or using vector loads would be better, but keeping it simple for correctness
        for (int i = 0; i < height * width; ++i) {
            spatial_sum += channel_ptr[i];
        }
        float mean_val = spatial_sum / (height * width);
        
        // 2. Add Bias
        float val = mean_val + bias_ptr[c];
        
        // 3. Accumulate for LogSumExp
        // We need to find the max and sum exp(val - max)
        // Since we are iterating over channels, we can update local max and sum_exp
        // But we need the global max to compute the final sum.
        // We can't compute exp(val - local_max) yet because local_max might not be the global max.
        // We can just store the 'val' in shared memory? No, shared memory is limited.
        // Better approach: Update local max. 
        // But we need to compute exp(x - global_max).
        // Standard parallel reduction approach:
        // Pass 1: Find global max.
        // Pass 2: Compute sum(exp(x - global_max)).
        // This requires reading the data twice or storing intermediate values.
        // Given the data size (C channels), we can store the 'val' in registers or shared memory if small.
        // But C can be large.
        // Optimization: Online algorithm for LogSumExp.
        // Maintain (max, sum) pair.
        // If new value x comes in:
        //   if x > max: sum = sum * exp(max - x) + 1; max = x;
        //   else: sum = sum + exp(x - max);
        // This works perfectly for the loop over channels.
        
        if (val > local_max) {
            local_sum_exp = local_sum_exp * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum_exp += expf(val - local_max);
        }
    }

    // Store local results in shared memory for block reduction
    // We need to synchronize, so we use shared memory.
    s_data[threadIdx.x] = local_max;
    s_data[threadIdx.x + blockDim.x] = local_sum_exp;
    __syncthreads();

    // Now perform reduction in shared memory to combine results from all threads
    // We need to combine (max, sum) pairs.
    // Binary reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float max1 = s_data[threadIdx.x];
            float sum1 = s_data[threadIdx.x + blockDim.x];
            float max2 = s_data[threadIdx.x + s];
            float sum2 = s_data[threadIdx.x + s + blockDim.x];

            // Combine two LogSumExp states
            // We want to compute log(exp(sum1) * exp(max1) + exp(sum2) * exp(max2)) 
            // Wait, the state is (max, sum_exp) where sum_exp = Sum(exp(x - max)).
            // The value represented is max + log(sum_exp).
            // We want to merge two such states.
            // Let M = max(max1, max2).
            // New Sum = Sum1 * exp(max1 - M) + Sum2 * exp(max2 - M).
            // New Max = M.
            
            float new_max = fmaxf(max1, max2);
            float new_sum_exp = sum1 * expf(max1 - new_max) + sum2 * expf(max2 - new_max);
            
            s_data[threadIdx.x] = new_max;
            s_data[threadIdx.x + blockDim.x] = new_sum_exp;
        }
        __syncthreads();
    }

    // Thread 0 writes the final result
    if (threadIdx.x == 0) {
        float final_max = s_data[0];
        float final_sum_exp = s_data[blockDim.x];
        float logsumexp_val = final_max + logf(final_sum_exp);
        
        // 4. Sum over spatial (already 1x1) and 5. Multiply by 10.0
        output[n] = logsumexp_val * 10.0f;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias) {
    // Input dimensions: (N, C, H, W)
    // Bias dimensions: (C, 1, 1)
    // Output dimensions: (N, 1)
    
    int batch_size = input.size(0);
    int out_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    // Launch configuration
    // One block per batch item
    int threads_per_block = 256; // Should be >= out_channels if possible, or handle striding
    // Shared memory size: 2 * threads_per_block * sizeof(float)
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    fused_ops_kernel<<<batch_size, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        height,
        width
    );
    
    return output;
}
"""

custom_kernel_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=custom_kernel_cpp_source,
    cuda_sources=custom_kernel_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Transposed Convolution (using PyTorch's optimized implementation)
        x = self.conv_transpose(x)
        
        # 2. Fused Custom CUDA Kernel for:
        # Global Average Pooling + Bias + LogSumExp + Sum + Multiplication
        x = self.fused_ops.fused_ops_cuda(x, self.bias)
        
        return x