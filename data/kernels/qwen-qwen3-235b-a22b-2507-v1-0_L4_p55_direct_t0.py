import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused matmul + maxpool1d + sum + scale operation
fused_kernel_source = """
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
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (out_idx >= out_features) return;

    float sum_val = 0.0f;
    for (int i = 0; i < in_features; i++) {
        sum_val += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
    }
    sum_val += bias[out_idx];

    // Apply max pooling: treat output as sequence of length out_features, pool with kernel_size
    float max_val = -FLT_MAX;
    int pooled_size = (out_features + kernel_size - 1) / kernel_size;
    int pool_start = (out_idx / kernel_size) * kernel_size;
    int pool_end = min(pool_start + kernel_size, out_features);
    
    // Each thread computes one element of the pooled output
    if (out_idx < pooled_size * kernel_size) {
        // We use one thread per output channel after pooling
        int global_pool_idx = out_idx;
        int src_start = (global_pool_idx / kernel_size) * kernel_size;
        int src_end = min(src_start + kernel_size, out_features);
        float local_max = -FLT_MAX;
        for (int j = src_start; j < src_end; j++) {
            float val = (j == src_start) ? sum_val : 0.0f;
            // Only one thread will have j == out_idx to simulate the matmul result
            if (j == out_idx) {
                val = sum_val;
            }
            local_max = fmaxf(local_max, val);
        }
        // Now reduce within block to compute sum over pooled sequence
        extern __shared__ float sdata[];
        sdata[threadIdx.x] = local_max;
        __syncthreads();

        // Simple reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float final_max = sdata[0];
            float sum_reduced = 0.0f;
            // We need to sum over all pooled outputs — but this kernel is not ideal for that
            // So we switch strategy: compute full matmul, then maxpool, then sum in separate steps
        }
    }
}
"""

# Instead, we do a more practical fusion: matmul + row-wise max pooling + sum + scale in two steps
# Step 1: matmul + maxpool
# Step 2: sum + scale

matmul_maxpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel for fused linear (matmul + bias) + maxpool1d
__global__ void fused_matmul_maxpool_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* temp_output,
    int batch_size,
    int in_features,
    int out_features,
    int kernel_size
) {
    int batch_idx = blockIdx.x;
    int out_group = blockIdx.y;  // each group corresponds to a pooling window
    int tid = threadIdx.x;
    int pool_width = kernel_size;
    int out_per_group = pool_width;
    int total_groups = (out_features + pool_width - 1) / pool_width;

    if (out_group >= total_groups) return;

    int start_idx = out_group * out_per_group;
    int end_idx = min(start_idx + out_per_group, out_features);

    float local_max = -FLT_MAX;

    // Each thread processes one output dim in the group
    if (tid < end_idx - start_idx) {
        int out_idx = start_idx + tid;
        float sum_val = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum_val += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }
        sum_val += bias[out_idx];
        local_max = sum_val;
    }

    // Reduce within the block to get max over the group
    float* shared = shared_memory<float>();
    shared[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        temp_output[batch_idx * total_groups + out_group] = shared[0];
    }
}

// Explicit shared memory helper
template<typename T>
__device__ T* shared_memory() {
    extern __shared__ T shared[];
    return shared;
}

// Kernel for sum + scale
__global__ void sum_scale_kernel(
    const float* temp_output,
    float* output,
    int batch_size,
    int pooled_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    float sum_val = 0.0f;

    // Each block handles one batch
    for (int i = tid; i < pooled_size; i += blockDim.x) {
        sum_val += temp_output[batch_idx * pooled_size + i];
    }

    // Reduce within block
    float* shared = shared_memory<float>();
    shared[tid] = sum_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx] = shared[0] * scale_factor;
    }
}

// Host wrapper
torch::Tensor fused_matmul_maxpool_sum_scale(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor temp_output,
    torch::Tensor output,
    int kernel_size,
    float scale_factor
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    const int pooled_size = (out_features + kernel_size - 1) / kernel_size;

    const int threads_per_block = 256;

    // Launch matmul + maxpool
    dim3 grid1(batch_size, pooled_size);
    dim3 block1(min(threads_per_block, kernel_size));
    size_t shared_mem1 = block1.x * sizeof(float);
    fused_matmul_maxpool_kernel<<<grid1, block1, shared_mem1>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        temp_output.data_ptr<float>(),
        batch_size, in_features, out_features, kernel_size
    );

    // Launch sum + scale
    dim3 grid2(batch_size);
    dim3 block2(threads_per_block);
    size_t shared_mem2 = block2.x * sizeof(float);
    sum_scale_kernel<<<grid2, block2, shared_mem2>>>(
        temp_output.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, pooled_size, scale_factor
    );

    return output;
}
"""

cpp_source = """
torch::Tensor fused_matmul_maxpool_sum_scale(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor temp_output,
    torch::Tensor output,
    int kernel_size,
    float scale_factor
);
"""

# Compile the CUDA kernel
fused_op = load_inline(
    name="fused_matmul_maxpool_sum_scale",
    cpp_sources=cpp_source,
    cuda_sources=matmul_maxpool_source,
    functions=["fused_matmul_maxpool_sum_scale"],
    verbose=False,
    extra_cuda_cflags=["-O3"],
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
        batch_size = x.size(0)
        pooled_size = (self.out_features + self.kernel_size - 1) // self.kernel_size
        temp_output = torch.empty(batch_size, pooled_size, device=x.device, dtype=x.dtype)
        output = torch.empty(batch_size, device=x.device, dtype=x.dtype)
        return fused_op.fused_matmul_maxpool_sum_scale(
            x, self.weight, self.bias, temp_output, output,
            self.kernel_size, self.scale_factor
        )