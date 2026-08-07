import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused matmul + maxpool1d + sum + scale
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Fused kernel: Linear (matmul + bias) -> MaxPool1d -> Sum -> Scale
__global__ void fused_matmul_maxpool_sum_scale_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;

    // Shared memory for reductions
    extern __shared__ float shared[];

    float* shared_vals = shared;
    int threads_per_block = blockDim.x;
    int tid = threadIdx.x;

    float acc = 0.0f;

    // Each thread processes multiple input features for matmul
    for (int i = tid; i < in_features; i += threads_per_block) {
        acc += input[batch_idx * in_features + i];
    }

    // Reduce across threads to get sum of inputs (will scale weight later)
    for (int s = threads_per_block / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            acc += shared_vals[tid] = shared_vals[tid + s];
        }
    }
    __syncthreads();

    // Now compute full matmul: output[batch][j] = sum_i input[i] * weight[j][i] + bias[j]
    // We'll compute one output feature per thread block iteration
    float total_sum = 0.0f;
    for (int j_start = 0; j_start < out_features; j_start += threads_per_block) {
        int j = j_start + tid;
        float activation = 0.0f;
        if (j < out_features) {
            activation = bias[j];
            for (int i = 0; i < in_features; i++) {
                activation += input[batch_idx * in_features + i] * weight[j * in_features + i];
            }
        }

        // Store activation for max pooling
        shared_vals[tid] = activation;
        __syncthreads();

        // Perform max pooling in groups of kernel_size
        int group_id = j / kernel_size;
        int group_start = group_id * kernel_size;
        int group_end = min(group_start + kernel_size, out_features);

        float max_in_group = -FLT_MAX;
        for (int idx = group_start; idx < group_end; idx++) {
            if (idx < out_features) {
                float val = shared_vals[idx - group_start + (idx >= group_start + threads_per_block ? 0 : 0)];
                // Fix indexing: use relative index within shared array
                int rel_idx = idx - group_start;
                if (rel_idx < threads_per_block) {
                    val = shared_vals[rel_idx];
                } else {
                    // Reload if not in shared memory
                    if (j == tid) {
                        val = activation;
                    }
                    // This approach is flawed — switch to per-group sync
                }
                max_in_group = fmaxf(max_in_group, val);
            }
        }

        // Instead, do: after storing all activations, do pooling in a separate step
        // We change strategy: first compute all activations, then pool, then sum
    }

    // NEW STRATEGY: Two-phase kernel with proper tiling
}
"""

# Let's rewrite with a simpler, correct approach: avoid excessive shared memory use and do proper grid-stride loops
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_matmul_maxpool_sum_scale_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;

    // We'll compute the entire output for this batch element
    float sum_pooled = 0.0f;

    // Loop over output features in groups (for max pooling)
    for (int start = 0; start < out_features; start += kernel_size) {
        int end = min(start + kernel_size, out_features);
        float max_val = -FLT_MAX;

        // Compute matmul + bias for each feature in the pool window
        for (int j = start; j < end; j++) {
            float val = bias[j];
            for (int i = 0; i < in_features; i++) {
                val += input[batch_idx * in_features + i] * weight[j * in_features + i];
            }
            max_val = fmaxf(max_val, val);
        }

        sum_pooled += max_val;
    }

    output[batch_idx] = sum_pooled * scale_factor;
}
"""

cpp_source = """
torch::Tensor fused_op_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    float scale_factor
);
"""

# Compile the CUDA kernel
fused_op = load_inline(
    name="fused_matmul_maxpool_sum_scale",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_op_launcher"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for fused matmul, maxpool, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return fused_op.fused_op_launcher(
            x, self.weight, self.bias, self.kernel_size, self.scale_factor
        )