import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + GELU + softmax
fused_linear_gelu_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to fuse linear (matmul + bias), GELU, and softmax
__global__ void fused_linear_gelu_softmax_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= out_features) return;

    // Linear: compute dot product of input and weight + bias
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];

    // GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))
    float x = sum;
    float gelu = x * 0.5f * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));

    // Store GELU result in shared memory for softmax computation
    extern __shared__ float s_data[];
    s_data[threadIdx.x] = gelu;

    __syncthreads();

    // Softmax: subtract max for numerical stability
    float max_val = s_data[0];
    for (int i = 1; i < blockDim.x; i++) {
        max_val = fmaxf(max_val, s_data[i]);
    }

    // Compute exp and sum
    float exp_val = expf(s_data[threadIdx.x] - max_val);
    s_data[threadIdx.x] = exp_val;
    __syncthreads();

    float sum_exp = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        sum_exp += s_data[i];
    }

    // Normalize
    output[row * out_features + col] = exp_val / sum_exp;
}
"""

fused_linear_gelu_softmax_cpp = """
torch::Tensor fused_linear_gelu_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_features,
    int out_features);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_linear_gelu_softmax",
    cpp_sources=[fused_linear_gelu_softmax_cpp],
    cuda_sources=[fused_linear_gelu_softmax_source],
    functions=["fused_linear_gelu_softmax_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused linear + GELU + softmax in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.batch_size = 1024
        self.in_features = in_features
        self.out_features = out_features
        self.fused_op = fused_op

    def forward(self, x):
        batch_size = x.size(0)
        output = self.fused_op.fused_linear_gelu_softmax_cuda(
            x, self.linear.weight, self.linear.bias,
            batch_size, self.in_features, self.out_features
        )
        return output