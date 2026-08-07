import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants for GELU approximation
constexpr float GELU_COEF = 0.7978845608028654f; // sqrt(2.0 / M_PI)
constexpr float GELU_CONST = 0.044715f;

// Kernel for fused LogSumExp, LeakyReLU, LeakyReLU, GELU, GELU
__global__ void fused_op_kernel(const float* __restrict__ x, float* __restrict__ out, int batch_size, int out_features) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Use shared memory for reduction
    // We need to compute max and sum for LogSumExp
    extern __shared__ float s_data[];
    float* s_max = s_data;
    float* s_sum = s_data + blockDim.x;

    // 1. LogSumExp
    // Step 1: Find max in the row
    float local_max = -INFINITY;
    for (int col = tid; col < out_features; col += blockDim.x) {
        float val = x[row * out_features + col];
        if (val > local_max) local_max = val;
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) {
                s_max[tid] = s_max[tid + s];
            }
        }
        __syncthreads();
    }
    float row_max = s_max[0];
    __syncthreads();

    // Step 2: Compute sum(exp(x - max))
    float local_sum = 0.0f;
    for (int col = tid; col < out_features; col += blockDim.x) {
        local_sum += expf(x[row * out_features + col] - row_max);
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = s_sum[0];
    
    // Compute LogSumExp result
    float lse = logf(row_sum) + row_max;

    // 2. LeakyReLU (x2)
    // 3. GELU (x2)
    // Apply activation sequence to the single scalar result
    float val = lse;

    // LeakyReLU 1
    val = (val > 0) ? val : val * 0.01f;
    // LeakyReLU 2
    val = (val > 0) ? val : val * 0.01f;
    
    // GELU 1
    float k1 = GELU_COEF * (val + GELU_CONST * val * val * val);
    val = 0.5f * val * (1.0f + tanhf(k1));
    
    // GELU 2
    float k2 = GELU_COEF * (val + GELU_CONST * val * val * val);
    val = 0.5f * val * (1.0f + tanhf(k2));

    // Write result
    if (tid == 0) {
        out[row] = val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int out_features = x.size(1);
    
    // Output tensor shape (batch_size, 1)
    auto out = torch::empty({batch_size, 1}, x.options());

    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = 2 * block_size * sizeof(float);

    fused_op_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), batch_size, out_features
    );

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_op = fused_op

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        # Fused LogSumExp, LeakyReLU, LeakyReLU, GELU, GELU
        x = self.fused_op.fused_op_cuda(x)
        return x