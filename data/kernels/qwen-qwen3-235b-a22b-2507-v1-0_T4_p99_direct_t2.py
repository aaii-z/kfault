import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + GELU + softmax
fused_linear_gelu_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Approximate GELU using tanh approximation
__device__ float gelu_approx(float x) {
    float tanh_arg = 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
    return x * 0.5f * (1.0f + tanhf(tanh_arg));
}

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
    int col = threadIdx.x;

    if (row >= batch_size) return;

    extern __shared__ float s_data[];

    // Step 1: Linear transformation (fully connected)
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];

    // Step 2: GELU activation
    float gelu_out = gelu_approx(sum);
    s_data[col] = gelu_out;
    __syncthreads();

    // Step 3: Softmax within the feature dimension (out_features)
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < out_features; i++) {
        max_val = fmaxf(max_val, s_data[i]);
    }
    __syncthreads();

    // Compute exp and sum
    float exp_val = expf(s_data[col] - max_val);
    s_data[col] = exp_val;
    __syncthreads();

    float sum_exp = 0.0f;
    for (int i = 0; i < out_features; i++) {
        sum_exp += s_data[i];
    }
    __syncthreads();

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
    int out_features) {
    auto output = torch::empty({batch_size, out_features}, input.options());
    const int block_size = 1024;
    const int shared_mem_size = out_features * sizeof(float);
    dim3 grid(batch_size);
    dim3 block(out_features);

    if (out_features > block_size) {
        throw std::runtime_error("out_features exceeds maximum supported block size");
    }

    fused_linear_gelu_softmax_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }

    return output;
}
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
        self.in_features = in_features
        self.out_features = out_features
        self.fused_op = fused_op

    def forward(self, x):
        batch_size = x.size(0)
        return self.fused_op.fused_linear_gelu_softmax_cuda(
            x, self.linear.weight, self.linear.bias,
            batch_size, self.in_features, self.out_features
        )