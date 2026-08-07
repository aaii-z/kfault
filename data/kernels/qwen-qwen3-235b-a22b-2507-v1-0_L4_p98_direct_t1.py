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
    int out_idx = threadIdx.x;
    int tid = threadIdx.x;

    // Shared memory for input, matmul output, and pooled+activated values
    extern __shared__ float s_data[];
    float* s_input = s_data;
    float* s_output = s_data;
    float* s_pooled = s_data + out_features;

    // Load input row
    for (int i = tid; i < in_features; i += blockDim.x) {
        s_input[i] = input[batch_idx * in_features + i];
    }
    __syncthreads();

    // Perform matmul: output = input @ weight.T + bias
    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        sum += s_input[i] * weight[out_idx * in_features + i];
    }
    if (out_idx < out_features) {
        sum += bias[out_idx];
        s_output[out_idx] = sum;
    }
    __syncthreads();

    // Apply AvgPool1d: pool over groups of `pool_kernel_size`
    int pooled_size = out_features / pool_kernel_size;
    if (out_idx < pooled_size) {
        float avg = 0.0f;
        int start = out_idx * pool_kernel_size;
        int end = start + pool_kernel_size;
        for (int i = start; i < end; ++i) {
            avg += s_output[i];
        }
        avg /= pool_kernel_size;
        s_pooled[out_idx] = gelu_activation(avg) * scale_factor;
    }
    __syncthreads();

    // Block-wide max reduction using shared memory
    if (out_idx < pooled_size) {
        s_output[out_idx] = s_pooled[out_idx];
    } else {
        s_output[out_idx] = -FLT_MAX;
    }
    __syncthreads();

    // Reduce to find max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (out_idx < stride && out_idx + stride < pooled_size) {
            s_output[out_idx] = fmaxf(s_output[out_idx], s_output[out_idx + stride]);
        }
        __syncthreads();
    }

    // Write result
    if (out_idx == 0) {
        output[batch_idx] = s_output[0];
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