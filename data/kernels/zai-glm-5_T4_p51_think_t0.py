import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_op_kernel(const float* __restrict__ x, const float* __restrict__ w_sum, float bias_sum, int in_features, int out_features, float* __restrict__ out) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Pointers to the current row
    const float* x_row = x + row * in_features;
    float* out_row = out + row * in_features;

    // Shared memory for the input row and partial sums
    extern __shared__ float s_data[];
    float* s_x = s_data;
    float* s_partial = s_data + in_features; // Store partial sums after x

    // 1. Load x to shared memory
    for (int i = tid; i < in_features; i += block_size) {
        s_x[i] = x_row[i];
    }
    __syncthreads();

    // 2. Compute dot product (part of Mean(Linear(x) - sub))
    // We compute sum(x_row * w_sum)
    float sum = 0.0f;
    for (int i = tid; i < in_features; i += block_size) {
        sum += s_x[i] * w_sum[i];
    }
    s_partial[tid] = sum;
    __syncthreads();

    // 3. Block reduction for the dot product
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial[tid] += s_partial[tid + s];
        }
        __syncthreads();
    }

    // 4. Compute final scalar value (Thread 0 handles the scalar ops)
    __shared__ float s_val;
    if (tid == 0) {
        float dot_product = s_partial[0];
        // Mean = (Sum(x * W_sum) + Sum(bias - sub)) / out_features
        float mean_val = (dot_product + bias_sum) / out_features;
        
        // LogSumExp on a scalar is identity: log(exp(x)) = x
        // So we skip it.
        
        // GELU
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        float gelu_val = 0.5f * mean_val * (1.0f + erff(mean_val * 0.7071067811865475f));
        
        s_val = gelu_val;
    }
    __syncthreads();

    // 5. Residual Add and write output
    // out = x + gelu_val
    for (int i = tid; i < in_features; i += block_size) {
        out_row[i] = s_x[i] + s_val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor w_sum, torch::Tensor bias_sum, int out_features) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = batch_size;
    
    // Shared memory size: in_features for x + block_size for partial sums
    size_t shared_mem_size = in_features * sizeof(float) + block_size * sizeof(float);

    // Extract scalar bias_sum
    float bias_sum_val = bias_sum.item<float>();

    fused_op_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), 
        w_sum.data_ptr<float>(), 
        bias_sum_val, 
        in_features, 
        out_features, 
        out.data_ptr<float>()
    );

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor w_sum, torch::Tensor bias_sum, int out_features);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that replaces the sequence of operations with a single custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.fused_op = fused_op

    def forward(self, x):
        # Pre-compute the sum of weights along the output dimension
        # This allows us to compute the sum of the linear layer output efficiently
        # sum(Linear(x)) = x @ sum(W) + sum(bias)
        w_sum = torch.sum(self.gemm.weight, dim=0)
        
        # Pre-compute the sum of (bias - subtract)
        if self.gemm.bias is not None:
            bias_sum = torch.sum(self.gemm.bias - self.subtract)
        else:
            bias_sum = -torch.sum(self.subtract)

        # Call the custom fused CUDA kernel
        # The kernel performs:
        # 1. Implicit GEMM reduction (Mean)
        # 2. LogSumExp (Identity on scalar)
        # 3. GELU
        # 4. Residual Add
        return self.fused_op.fused_op_cuda(x, w_sum, bias_sum, self.out_features)