import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sum + max + mean + logsumexp with high parallelism
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Efficient GEMV with bias using shared memory and coalesced access
__global__ void fused_linear_sum_max_mean_logsumexp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float sdata[];

    float thread_sum = 0.0f;
    float thread_max = -1e30f;

    // Use multiple threads per row to compute partial dot products
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Each thread processes several input features
    for (int i = tid; i < in_features; i += num_threads) {
        float input_val = input[row * in_features + i];
        for (int j = 0; j < out_features; j++) {
            float w = weight[j * in_features + i];
            sdata[j * in_features + i] = input_val * w;  // Store products
        }
    }

    __syncthreads();

    // Now reduce over out_features in parallel
    if (tid == 0) {
        for (int j = 0; j < out_features; j++) {
            float val = bias[j];
            for (int i = 0; i < in_features; i++) {
                val += sdata[j * in_features + i];
            }
            thread_sum += val;
            thread_max = fmaxf(thread_max, val);
        }
    }

    __syncthreads();

    // Only thread 0 continues
    if (tid != 0) return;

    float mean_val = thread_sum / out_features;

    // First logsumexp: log(sum(exp(x))) = max_val + log(sum(exp(x - max_val)))
    float acc = 0.0f;
    for (int j = 0; j < out_features; j++) {
        float val = bias[j];
        for (int i = 0; i < in_features; i++) {
            val += sdata[j * in_features + i];
        }
        acc += expf(val - thread_max);
    }
    float logsumexp1 = thread_max + logf(acc);

    // Second logsumexp: over a single element -> equals the element
    float logsumexp2 = logsumexp1;

    output[row] = logsumexp2;
}

torch::Tensor fused_linear_sum_max_mean_logsumexp(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = bias.size(0);

    auto output = torch::empty(batch_size, input.options());

    const int block_size = 256;
    const int num_blocks = batch_size;

    // Shared memory to store intermediate products: out_features * in_features
    size_t shared_mem_size = out_features * in_features * sizeof(float);

    fused_linear_sum_max_mean_logsumexp_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output.unsqueeze(-1);
}
"""

fused_cpp_source = """
torch::Tensor fused_linear_sum_max_mean_logsumexp(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_linear_sum_max_mean_logsumexp",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_sum_max_mean_logsumexp"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel replacing:
        - Linear (matmul + bias)
        - sum(dim=1)
        - max(dim=1)
        - mean(dim=1)
        - logsumexp(dim=1)
        - logsumexp(dim=1)
    All fused into a single CUDA kernel with high parallelism and coalesced memory access.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_linear_sum_max_mean_logsumexp(x, self.linear.weight, self.linear.bias)