import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias add, dropout, and softmax
fused_bias_dropout_softmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <limits>

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

// Simple XORWOW RNG state struct
struct RngState {
    unsigned int d, a, b, c;
};

// Initialize RNG state
__device__ void init_rng(RngState* state, unsigned long long seed, unsigned long long seq) {
    state->a = 0xF1EA5EED;
    state->b = seed ^ seq;
    state->c = seed ^ (seq >> 32);
    state->d = seed;
    for (int i = 0; i < 20; ++i) {
        unsigned int t = state->a ^ (state->a >> 7);
        state->a = state->b;
        state->b = state->c;
        state->c = state->d;
        state->d = (state->d ^ (state->d << 3)) ^ (t ^ (t << 6));
    }
}

// Generate random number in [0, 1)
__device__ float rng_uniform(RngState* state) {
    unsigned int t = state->a ^ (state->a >> 7);
    state->a = state->b;
    state->b = state->c;
    state->c = state->d;
    state->d = (state->d ^ (state->d << 3)) ^ (t ^ (t << 6));
    return (float)state->d / (float)UINT_MAX;
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

    // Dynamic shared memory for the row data
    extern __shared__ float s_data[];
    
    // Pointers to the current row
    const float* row_input = input + row_idx * features;
    float* row_output = output + row_idx * features;

    // 1. Load data, add bias, apply dropout, store in shared memory
    // Initialize RNG for this thread
    RngState rng_state;
    if (is_training && dropout_p > 0.0f) {
        init_rng(&rng_state, seed, row_idx * features + tid);
    }

    const float scale = 1.0f / (1.0f - dropout_p);

    for (int i = tid; i < features; i += block_dim) {
        float val = row_input[i] + bias[i];
        
        if (is_training && dropout_p > 0.0f) {
            float rand = rng_uniform(&rng_state);
            if (rand < dropout_p) {
                val = 0.0f;
            } else {
                val *= scale;
            }
        }
        s_data[i] = val;
    }
    __syncthreads();

    // 2. Find Max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < features; i += block_dim) {
        local_max = fmaxf(local_max, s_data[i]);
    }
    
    // Block reduction for max
    // Use shared memory for reduction across warps
    __shared__ float s_max[32]; // Enough for up to 1024 threads
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float warp_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
        s_max[warp_id] = warp_max;
    }
    __syncthreads();

    // First warp reduces the results
    if (warp_id == 0) {
        float block_max = (tid < (blockDim.x + 31) / 32) ? s_max[lane_id] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (tid == 0) {
            s_max[0] = block_max;
        }
    }
    __syncthreads();
    float row_max = s_max[0];

    // 3. Compute Exp and Sum
    float local_sum = 0.0f;
    for (int i = tid; i < features; i += block_dim) {
        float val = expf(s_data[i] - row_max);
        s_data[i] = val; // Store exp result back to shared mem
        local_sum += val;
    }

    // Block reduction for sum
    __shared__ float s_sum[32];
    float warp_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        s_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (tid < (blockDim.x + 31) / 32) ? s_sum[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            s_sum[0] = block_sum;
        }
    }
    __syncthreads();
    float row_sum = s_sum[0];

    // 4. Normalize and Write Output
    const float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < features; i += block_dim) {
        row_output[i] = s_data[i] * inv_sum;
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
    
    // Shared memory size: one row of floats
    const int shared_mem_size = features * sizeof(float);

    // Generate a random seed
    unsigned long long seed = 0;
    if (is_training) {
        seed = std::random_device{}();
    }

    fused_bias_dropout_softmax_kernel<<<blocks, threads, shared_mem_size>>>(
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
        # We keep the Linear layer but we will not use its forward pass directly
        # in the fused kernel. We use its weights.
        # We set bias=False in Linear because we handle it manually in the fused kernel.
        # However, to match the original architecture's parameters exactly (so we can load weights if needed),
        # we can keep bias=True in Linear and just pass self.matmul.bias to the kernel.
        # Let's stick to the standard definition to ensure compatibility.
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Matrix Multiplication (x @ W.T)
        # We perform the matmul using PyTorch's optimized GEMM
        # Note: self.matmul(x) does x @ W.T + b. 
        # We want to separate the bias addition to fuse it.
        # So we do x @ self.matmul.weight.T
        x = F.linear(x, self.matmul.weight, None)
        
        # 2. Fused Bias + Dropout + Softmax
        # We pass the bias from the Linear layer
        return self.fused_ops.fused_bias_dropout_softmax_cuda(x, self.matmul.bias, self.dropout_p, self.training)