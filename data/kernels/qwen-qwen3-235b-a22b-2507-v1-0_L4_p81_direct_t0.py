import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + Swish + Divide + Clamp + Tanh + Clamp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void fused_gemm_swish_divide_clamp_tanh_clamp(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= out_features) return;

    // GEMM: compute dot product of input[row, :] and weight[col, :]
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }

    // Add bias
    sum += bias[col];

    // Swish: x * sigmoid(x)
    float swish = sum * (1.0f / (1.0f + expf(-sum)));

    // Divide by 2.0
    float divided = swish / 2.0f;

    // First clamp: [-1.0, 1.0]
    float clamped1 = fmaxf(-1.0f, fminf(1.0f, divided));

    // Tanh
    float tanhed = tanhf(clamped1);

    // Second clamp: [-1.0, 1.0] (redundant for tanh but kept as per original)
    float result = fmaxf(-1.0f, fminf(1.0f, tanhed));

    output[row * out_features + col] = result;
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    const int grid_x = (out_features + block_size - 1) / block_size;
    dim3 grid(grid_x, batch_size);
    dim3 block(block_size);

    fused_gemm_swish_divide_clamp_tanh_clamp<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    cudaError_t err = cudaGetLastError();
    C10_CUDA_CHECK(err);

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized version of Model with fused GEMM + Swish + Divide + Clamp + Tanh + Clamp in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_op = fused_ops.fused_forward_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias)