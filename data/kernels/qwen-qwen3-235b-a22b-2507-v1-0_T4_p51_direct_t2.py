import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and Residual Add
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation approximation
__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

extern "C" __global__ void fused_gemm_subtract_globalavgpool_logsumexp_gelu_residual_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* subtract,
    float* output,
    int batch_size,
    int in_features,
    int out_features) {

    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Each block handles one batch sample
    if (bid >= batch_size) return;

    float thread_max = -1e30f;
    float thread_sum = 0.0f;

    // Step 1: Compute GEMM + Subtract for this batch sample
    for (int i = tid; i < out_features; i += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[bid * in_features + k] * weight[i * in_features + k];
        }
        if (bias != nullptr) {
            sum += bias[i];
        }
        sum -= subtract[i];
        s_data[i] = sum;
        thread_max = fmaxf(thread_max, sum);
    }

    __syncthreads();

    // Step 2: Block-wide max reduction using shared memory
    if (tid == 0) {
        float block_max = -1e30f;
        for (int i = 0; i < out_features; i++) {
            block_max = fmaxf(block_max, s_data[i]);
        }
        s_data[0] = block_max;
    }
    __syncthreads();
    float max_val = s_data[0];

    __syncthreads();

    // Step 3: Compute exp(x - max) and sum
    float exp_sum = 0.0f;
    for (int i = tid; i < out_features; i += blockDim.x) {
        exp_sum += expf(s_data[i] - max_val);
    }

    // Reduce exp_sum across threads using shared memory
    s_data[tid] = exp_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    float total_exp_sum = s_data[0];

    // Step 4: Compute logsumexp = log(total_exp_sum) + max_val
    float logsumexp_val = logf(total_exp_sum) + max_val;

    // Step 5: Apply GELU
    float gelu_val = gelu_activation(logsumexp_val);

    // Step 6: Residual add: output[bid, :] = gelu_val + input[bid, :]
    for (int i = tid; i < in_features; i += blockDim.x) {
        output[bid * in_features + i] = gelu_val + input[bid * in_features + i];
    }
}
"""

fused_cpp_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract);
"""

fused_wrapper_source = """
torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, in_features}, input.options());

    const int block_size = 256;
    const int grid_size = batch_size;

    // Shared memory for intermediate values (size: out_features * sizeof(float))
    size_t smem_size = out_features * sizeof(float);

    void* args[] = {
        (void*)&input.data_ptr<float>()[0],
        (void*)&weight.data_ptr<float>()[0],
        (void*)(bias.defined() ? bias.data_ptr<float>() : nullptr),
        (void*)&subtract.data_ptr<float>()[0],
        (void*)&output.data_ptr<float>()[0],
        (void*)&batch_size,
        (void*)&in_features,
        (void*)&out_features
    };

    cudaLaunchKernel(
        (void*)fused_gemm_subtract_globalavgpool_logsumexp_gelu_residual_kernel,
        dim3(grid_size), dim3(block_size), args, smem_size, 0
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=[fused_cpp_source, fused_wrapper_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_op_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.fused_op = fused_ops.fused_op_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias, self.subtract)