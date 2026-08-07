import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation for fused Matmul + AvgPool1d + GELU + Scale + Max
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__device__ float gelu_activation(float x) {
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__global__ void fused_matmul_avgpool_gelu_scale_max_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int pool_kernel_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;

    // Shared memory for matmul result and pooled/activated values
    extern __shared__ float s_data[];
    float* s_matmul = s_data;
    float* s_pooled = s_data + out_features;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Each thread loads multiple elements for matmul
    float sum = 0.0f;
    for (int i = tid; i < in_features; i += num_threads) {
        sum += input[batch_idx * in_features + i] * weight[threadIdx.x * in_features + i];
    }

    // Reduce within block to compute one element of matmul
    __shared__ float s_reduce[256];
    s_reduce[tid] = sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride && threadIdx.x < out_features) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the full dot product result
    if (threadIdx.x < out_features && tid == 0) {
        s_matmul[threadIdx.x] = s_reduce[0] + bias[threadIdx.x];
    }
    __syncthreads();

    // Perform AvgPool1d: pool over non-overlapping windows
    int pooled_size = out_features / pool_kernel_size;
    float local_max = -FLT_MAX;

    for (int pool_idx = tid; pool_idx < pooled_size; pool_idx += num_threads) {
        int start = pool_idx * pool_kernel_size;
        int end = start + pool_kernel_size;
        float avg = 0.0f;
        for (int i = start; i < end; ++i) {
            avg += s_matmul[i];
        }
        avg /= pool_kernel_size;
        float activated = gelu_activation(avg) * scale_factor;
        s_pooled[pool_idx] = activated;
        local_max = fmaxf(local_max, activated);
    }
    __syncthreads();

    // Store local max in shared memory
    s_reduce[tid] = local_max;
    __syncthreads();

    // Reduce across threads to get max
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        output[batch_idx] = s_reduce[0];
    }
}
"""

fused_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t pool_kernel_size,
    float scale_factor
);
"""

# Corrected CUDA kernel with proper reduction and indexing
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__device__ float gelu_activation(float x) {
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__global__ void fused_matmul_avgpool_gelu_scale_max_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int pool_kernel_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory: first out_features for matmul result, then pooled_size for pooled values
    extern __shared__ float s_data[];
    float* s_matmul = s_data;
    float* s_pooled = s_data + out_features;

    // Initialize shared memory
    for (int i = tid; i < out_features; i += block_size) {
        float sum = 0.0f;
        for (int j = 0; j < in_features; ++j) {
            sum += input[batch_idx * in_features + j] * weight[i * in_features + j];
        }
        s_matmul[i] = sum + bias[i];
    }
    __syncthreads();

    int pooled_size = out_features / pool_kernel_size;

    // AvgPool + GELU + Scale
    for (int i = tid; i < pooled_size; i += block_size) {
        float avg = 0.0f;
        int start = i * pool_kernel_size;
        for (int j = 0; j < pool_kernel_size; ++j) {
            avg += s_matmul[start + j];
        }
        avg /= pool_kernel_size;
        s_pooled[i] = gelu_activation(avg) * scale_factor;
    }
    __syncthreads();

    // Max reduction
    float max_val = -FLT_MAX;
    for (int i = tid; i < pooled_size; i += block_size) {
        max_val = fmaxf(max_val, s_pooled[i]);
    }

    // Use shared memory for reduction
    __shared__ float s_max[256];
    s_max[tid] = max_val;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx] = s_max[0];
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t pool_kernel_size,
    float scale_factor
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size}, input.options());

    const int block_size = 256;
    int pooled_size = out_features / pool_kernel_size;
    int shared_mem_size = (out_features + pooled_size) * sizeof(float);

    dim3 grid(batch_size);
    dim3 block(block_size);

    fused_matmul_avgpool_gelu_scale_max_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        static_cast<int>(pool_kernel_size),
        scale_factor
    );

    return output;
}
"""

fused_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t pool_kernel_size,
    float scale_factor
);
"""

fused_op = load_inline(
    name="fused_op",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused CUDA kernel for Matmul + AvgPool1d + GELU + Scale + Max.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(
            x,
            self.linear.weight,
            self.linear.bias,
            self.pool_kernel_size,
            self.scale_factor
        )