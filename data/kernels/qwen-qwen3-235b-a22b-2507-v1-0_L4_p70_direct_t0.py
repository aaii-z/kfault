import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Gemm_Sigmoid_Scaling_ResidualAdd
fused_gemm_sigmoid_scaling_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_gemm_sigmoid_scaling_residual_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float scaling_factor,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= hidden_size) return;

    // Compute GEMM: output[row][col] = sum_i (input[row][i] * weight[col][i]) + bias[col]
    float sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
        sum += input[row * input_size + i] * weight[col * input_size + i];
    }
    sum += bias[col];

    float original = sum;

    // Sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    sum = 1.0f / (1.0f + expf(-sum));

    // Scale
    sum = sum * scaling_factor;

    // Residual add
    sum = sum + original;

    output[row * hidden_size + col] = sum;
}

torch::Tensor fused_gemm_sigmoid_scaling_residual_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = weight.size(0);

    auto output = torch::empty({batch_size, hidden_size}, input.options());

    const int block_size = 256;
    dim3 block_dim(block_size);
    dim3 grid_dim((hidden_size + block_size - 1) / block_size, batch_size);

    fused_gemm_sigmoid_scaling_residual_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        input_size,
        hidden_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}
"""

fused_gemm_sigmoid_scaling_residual_cpp_source = """
torch::Tensor fused_gemm_sigmoid_scaling_residual_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_gemm_sigmoid_scaling_residual",
    cpp_sources=fused_gemm_sigmoid_scaling_residual_cpp_source,
    cuda_sources=fused_gemm_sigmoid_scaling_residual_source,
    functions=["fused_gemm_sigmoid_scaling_residual_cuda"],
    verbose=False,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Gemm_Sigmoid_Scaling_ResidualAdd in a single CUDA kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op.fused_gemm_sigmoid_scaling_residual_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias, self.scaling_factor)