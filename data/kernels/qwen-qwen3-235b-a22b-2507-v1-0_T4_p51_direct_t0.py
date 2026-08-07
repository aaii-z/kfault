import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and Residual Add
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_gemm_subtract_globalavgpool_logsumexp_gelu_residual_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* subtract,
    float* output,
    int batch_size,
    int in_features,
    int out_features) {

    int total_elements = batch_size * out_features;
    int linear_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for reduction in logsumexp and global avg pool
    extern __shared__ float s_data[];

    float thread_sum = 0.0f;
    float thread_max = -1e30f;

    // Gemm + Subtract in one pass per row
    for (int i = linear_id; i < total_elements; i += blockDim.x * gridDim.x) {
        int batch_idx = i / out_features;
        int out_idx = i % out_features;

        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[batch_idx * in_features + k] * weight[out_idx * in_features + k];
        }
        if (bias) {
            sum += bias[out_idx];
        }
        sum -= subtract[out_idx];  // Subtract parameter

        // Store intermediate result for next steps
        s_data[i] = sum;

        // Update max for logsumexp
        thread_max = fmaxf(thread_max, sum);
    }

    // Synchronize to ensure all writes to shared memory are complete
    __syncthreads();

    // Reduction to find global max
    float max_val = thread_max;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            max_val = fmaxf(max_val, s_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    s_data[threadIdx.x] = max_val;
    __syncthreads();
    max_val = s_data[0];

    // Compute exp(x - max) and sum for logsumexp
    float exp_sum = 0.0f;
    for (int i = linear_id; i < total_elements; i += blockDim.x * gridDim.x) {
        float exp_val = expf(s_data[i] - max_val);
        exp_sum += exp_val;
    }

    // Reduction for exp_sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        exp_sum += __shfl_down_sync(0xffffffff, exp_sum, stride);
    }

    // First thread in block has the total exp_sum
    float logsumexp_val = (threadIdx.x == 0) ? (logf(exp_sum) + max_val) : 0.0f;
    __syncthreads();
    if (threadIdx.x == 0) {
        s_data[0] = logsumexp_val;
    }
    __syncthreads();
    logsumexp_val = s_data[0];

    // Apply GELU (scalar operation on logsumexp result)
    float gelu_val = gelu_activation(logsumexp_val);

    // Write final result: residual add (gelu_val + original input's first value? but shape mismatch)
    // However, original code does: x = x + original_x, where x is (B, 1) and original_x is (B, in_features)
    // This is invalid unless original_x is reduced. But in the original model, after logsumexp and GELU, x is (B, 1)
    // and original_x is (B, in_features) -> cannot add directly.

    // Let's re-analyze: 
    // After GlobalAvgPool: x is (B, 1)
    // After LogSumExp: x is (B, 1)
    // After GELU: x is (B, 1)
    // Then: x + original_x -> (B, 1) + (B, in_features) -> broadcasting fails unless in_features == 1.

    // But in_features = 8192. So this must be a bug? Or perhaps GlobalAvgPool is over wrong dim?

    // Looking back: 
    # x = torch.mean(x, dim=1, keepdim=True)  --> x: [B, out_features] -> [B, 1]
    # Then logsumexp over dim=1 -> [B, 1]
    # Then GELU -> [B, 1]
    # Then x + original_x: [B, 1] + [B, 8192] -> PyTorch will broadcast [B,1] to [B,8192] and add.

    // So the residual connection broadcasts the scalar (per batch) to all in_features.

    // Therefore, we need to output a tensor of shape (B, in_features), where each row is the scalar (gelu_val) + original input.

    // So final output per batch: output[batch_idx * in_features + k] = gelu_val + input[batch_idx * in_features + k]

    for (int i = linear_id; i < batch_size * in_features; i += blockDim.x * gridDim.x) {
        output[i] = gelu_val + input[i];
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor subtract) {

    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::zeros({batch_size, in_features}, input.options());

    const int block_size = 256;
    const int num_blocks = 1; // We use grid-stride loops, so 1 block is enough if we use cooperative groups, but let's use more
    // Actually, we need to cover batch_size * out_features in first phase, so better use more blocks
    const int phase1_blocks = (batch_size * out_features + block_size - 1) / block_size;
    const int phase2_blocks = (batch_size * in_features + block_size - 1) / block_size;

    // We'll use dynamic shared memory for reduction
    size_t smem_size = block_size * sizeof(float);

    fused_gemm_subtract_globalavgpool_logsumexp_gelu_residual_kernel<<<phase2_blocks, block_size, smem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor subtract);
"""

# Compile the inline CUDA kernel
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
        self.fused_op = fused_ops.fused_op_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias, self.subtract)