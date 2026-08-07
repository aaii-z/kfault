import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and Residual Add
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

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

    int total_elements = batch_size * out_features;
    float thread_max = -1e30f;
    float thread_sum_exp = 0.0f;

    // Step 1: GEMM + Subtract - compute per (batch, out_feature) value
    CUDA_1D_KERNEL_LOOP(i, total_elements) {
        int batch_idx = i / out_features;
        int out_idx = i % out_features;

        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[batch_idx * in_features + k] * weight[out_idx * in_features + k];
        }
        if (bias != nullptr) {
            sum += bias[out_idx];
        }
        sum -= subtract[out_idx];
        s_data[i] = sum;
        thread_max = fmaxf(thread_max, sum);
    }
    __syncthreads();

    // Step 2: Block-wide max reduction
    float* shared_max = s_data;
    shared_max[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();

    // Step 3: Compute exp(x - max) and accumulate sum
    float local_exp_sum = 0.0f;
    CUDA_1D_KERNEL_LOOP(i, total_elements) {
        local_exp_sum += expf(s_data[i] - max_val);
    }

    // Block-wide sum reduction for exp sum
    float* shared_sum = s_data;
    shared_sum[tid] = local_exp_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float total_exp_sum = shared_sum[0];
    __syncthreads();

    // Step 4: LogSumExp and GELU
    float logsumexp_val = logf(total_exp_sum) + max_val;
    float gelu_val = gelu_activation(logsumexp_val);

    // Step 5: Residual Add - Add original input back
    CUDA_1D_KERNEL_LOOP(i, batch_size * in_features) {
        output[i] = gelu_val + input[i];
    }
}

extern "C" void fused_gemm_subtract_globalavgpool_logsumexp_gelu_residual(
    const float* input,
    const float* weight,
    const float* bias,
    const float* subtract,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream) {

    const int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = 1; // Only one block to allow full reduction

    int smem_size = block_size * sizeof(float);

    fused_gemm_subtract_globalavgpool_logsumexp_gelu_residual_kernel<<<num_blocks, block_size, smem_size, stream>>>(
        input, weight, bias, subtract, output,
        batch_size, in_features, out_features
    );
}
"""

fused_cpp_source = """
#include <torch/extension.h>

void fused_gemm_subtract_globalavgpool_logsumexp_gelu_residual(
    const float* input,
    const float* weight,
    const float* bias,
    const float* subtract,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream);

at::Tensor fused_op_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor subtract) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, in_features}, input.options());

    fused_gemm_subtract_globalavgpool_logsumexp_gelu_residual(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        at::cuda::getCurrentCUDAStream()
    );

    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return fused_ops.fused_op_cuda(x, self.linear.weight, self.linear.bias, self.subtract)