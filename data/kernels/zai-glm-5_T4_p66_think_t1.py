import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias add, dropout, and softmax
# Using an online softmax algorithm to avoid storing the whole row in shared memory,
# which allows handling large feature dimensions (e.g., 16384) without shared memory overflow.
fused_bias_dropout_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <random>

// Helper function for warp-level reduction (Sum)
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Helper function for warp-level reduction (Max)
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Simple device RNG to generate uniform random numbers
__device__ float get_random_uniform(unsigned long long seed, int idx) {
    // Simple LCG
    unsigned long long state = seed + idx;
    state = (state * 1103515245ULL + 12345ULL) & 0x7fffffffULL;
    return (float)state / (float)0x7fffffff;
}

__global__ void fused_bias_dropout_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int features,
    const float dropout_p,
    const bool is_training,
    const unsigned long long seed) {

    // One block per row (batch item)
    const int row_idx = blockIdx.x;
    if (row_idx >= batch_size) return;

    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;

    // Pointers to the current row
    const float* row_input = input + row_idx * features;
    float* row_output = output + row_idx * features;

    // 1. First pass: Compute Max and Sum (Online Softmax)
    // We iterate through the row in chunks.
    // thread_max and thread_sum accumulate values for the strided elements handled by this thread.
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    
    const float scale = 1.0f / (1.0f - dropout_p);

    for (int i = tid; i < features; i += block_dim) {
        float val = row_input[i] + bias[i];
        
        if (is_training && dropout_p > 0.0f) {
            float rand = get_random_uniform(seed, row_idx * features + i);
            if (rand < dropout_p) {
                val = 0.0f;
            } else {
                val *= scale;
            }
        }
        
        // Online softmax update
        float new_max = fmaxf(thread_max, val);
        // Rescale previous sum: exp(thread_max - new_max) * thread_sum
        // Add new element: exp(val - new_max)
        // If new_max is infinity, handle gracefully (though unlikely for initialized weights)
        if (new_max > -INFINITY) {
            float prev_scale = expf(thread_max - new_max);
            float curr_val = expf(val - new_max);
            thread_sum = prev_scale * thread_sum + curr_val;
            thread_max = new_max;
        }
    }

    // Block reduction for max and sum
    __shared__ float s_max[32]; // Enough for up to 1024 threads
    __shared__ float s_sum[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float warp_max = warp_reduce_max(thread_max);
    float warp_sum = warp_reduce_sum(thread_sum);

    if (lane_id == 0) {
        s_max[warp_id] = warp_max;
        s_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    // First warp reduces the results
    if (warp_id == 0) {
        float block_max = (tid < (blockDim.x + 31) / 32) ? s_max[lane_id] : -INFINITY;
        float block_sum = (tid < (blockDim.x + 31) / 32) ? s_sum[lane_id] : 0.0f;
        
        block_max = warp_reduce_max(block_max);
        block_sum = warp_reduce_sum(block_sum);
        
        if (tid == 0) {
            s_max[0] = block_max;
            s_sum[0] = block_sum;
        }
    }
    __syncthreads();
    
    float row_max = s_max[0];
    float row_sum = s_sum[0];
    float inv_sum = 1.0f / row_sum;

    // 2. Second pass: Normalize and Write Output
    // We need to re-compute the values because we didn't store them.
    // This is compute-heavy but memory-light, suitable for large features.
    for (int i = tid; i < features; i += block_dim) {
        float val = row_input[i] + bias[i];
        
        if (is_training && dropout_p > 0.0f) {
            float rand = get_random_uniform(seed, row_idx * features + i);
            if (rand < dropout_p) {
                val = 0.0f;
            } else {
                val *= scale;
            }
        }
        
        row_output[i] = expf(val - row_max) * inv_sum;
    }
}

torch::Tensor fused_bias_dropout_softmax_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    double dropout_p,
    bool is_training) {

    auto batch_size = input.size(0);
    auto features = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = batch_size;

    // Generate a random seed
    unsigned long long seed = 0;
    if (is_training) {
        // Use random_device for seed generation
        seed = std::random_device{}();
    }

    fused_bias_dropout_softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        static_cast<float>(dropout_p),
        is_training,
        seed
    );

    return output;
}
"""

fused_bias_dropout_softmax_cpp_source = (
    "torch::Tensor fused_bias_dropout_softmax_cuda(torch::Tensor input, torch::Tensor bias, double dropout_p, bool is_training);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_bias_dropout_softmax_cpp_source,
    cuda_sources=fused_bias_dropout_softmax_source,
    functions=["fused_bias_dropout_softmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused bias, dropout, and softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Matrix Multiplication (x @ W.T)
        # We perform the matmul using PyTorch's optimized GEMM, but separate bias addition.
        x = F.linear(x, self.matmul.weight, None)
        
        # 2. Fused Bias + Dropout + Softmax
        # We pass the bias from the Linear layer
        return self.fused_ops.fused_bias_dropout_softmax_cuda(x, self.matmul.bias, self.dropout_p, self.training)