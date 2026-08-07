import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused matmul + maxpool1d + sum + scale
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Shared memory helper
template<typename T>
__device__ T* shared_memory() {
    extern __shared__ T shared[];
    return shared;
}

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
    int pooled_out_features = (out_features + kernel_size - 1) / kernel_size;
    int threads_per_block = blockDim.x;
    int tid = threadIdx.x;

    // Shared memory for reduction in both max pooling and final sum
    extern __shared__ float shared_data[];
    float* shared_max = shared_data;
    float* shared_sum = shared_data + threads_per_block;

    float sum_val = 0.0f;

    // Step 1: Compute matmul + bias for all output features
    // We'll compute all out_features values, but only keep max per pool
    float* act = shared_max; // Reuse shared memory to store activations for this block

    for (int out_idx = tid; out_idx < out_features; out_idx += threads_per_block) {
        float val = bias[out_idx];
        for (int i = 0; i < in_features; i++) {
            val += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }
        if (out_idx < out_features) {
            act[out_idx] = val;
        }
    }
    __syncthreads();

    // Step 2: Max pooling within each kernel_size group
    int pooled_val_idx = 0;
    float* pooled_vals = shared_sum;
    for (int group_start = 0; group_start < out_features; group_start += kernel_size) {
        float max_val = -FLT_MAX;
        int group_end = min(group_start + kernel_size, out_features);
        for (int j = group_start + tid; j < group_end; j += threads_per_block) {
            max_val = fmaxf(max_val, act[j]);
        }

        // Reduce within thread block to get max of this group
        shared_max[tid] = max_val;
        for (int s = threads_per_block / 2; s > 0; s >>= 1) {
            __syncthreads();
            if (tid < s) {
                shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
            }
        }
        __syncthreads();

        // Thread 0 writes the pooled value
        if (tid == 0 && pooled_val_idx < pooled_out_features) {
            pooled_vals[pooled_val_idx] = shared_max[0];
        }
        __syncthreads();
        pooled_val_idx++;
    }

    // Step 3: Sum all pooled values
    float local_sum = 0.0f;
    for (int i = tid; i < pooled_out_features; i += threads_per_block) {
        local_sum += pooled_vals[i];
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Final reduction for sum
    for (int s = threads_per_block / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes final scaled output
    if (tid == 0) {
        output[batch_idx] = shared_sum[0] * scale_factor;
    }
}

// C++ wrapper
torch::Tensor fused_op_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    float scale_factor
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size}, input.options());

    const int threads_per_block = 256;
    const int pooled_out_features = (out_features + kernel_size - 1) / kernel_size;
    const int smem_size = threads_per_block * 2 * sizeof(float); // for max and sum

    dim3 grid(batch_size);
    dim3 block(threads_per_block);

    fused_matmul_maxpool_sum_scale_kernel<<<grid, block, smem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        kernel_size,
        scale_factor
    );

    return output;
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