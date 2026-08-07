import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear, divide, and GELU operation
fused_linear_div_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define M_SQRT2 1.41421356237309504880f
#define M_2_SQRTPI 1.12837916709551257390f

__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(M_SQRT2 * M_2_SQRTPI * x * (1.0f + 0.044715f * x * x)));
}

__global__ void fused_linear_div_gelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_size,
    int output_size,
    float divisor
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= output_size) return;

    float sum = 0.0f;
    for (int k = 0; k < input_size; k++) {
        sum += input[row * input_size + k] * weight[col * input_size + k];
    }
    sum += bias[col];
    sum /= divisor;
    output[row * output_size + col] = gelu_activation(sum);
}

torch::Tensor fused_linear_div_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    int batch_size,
    int input_size,
    int output_size
) {
    auto output = torch::empty({batch_size, output_size}, input.options());

    const int block_size = 256;
    dim3 block_dim(block_size);
    dim3 grid_dim((output_size + block_size - 1) / block_size, batch_size);

    fused_linear_div_gelu_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size,
        divisor
    );

    return output;
}
"""

fused_linear_div_gelu_cpp_source = """
torch::Tensor fused_linear_div_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    int batch_size,
    int input_size,
    int output_size
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_linear_div_gelu",
    cpp_sources=fused_linear_div_gelu_cpp_source,
    cuda_sources=fused_linear_div_gelu_source,
    functions=["fused_linear_div_gelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused linear, divide, and GELU operation in a single CUDA kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.input_size = input_size
        self.output_size = output_size
        self.fused_op = fused_op

    def forward(self, x):
        batch_size = x.size(0)
        return self.fused_op.fused_linear_div_gelu_cuda(
            x, self.linear.weight, self.linear.bias, self.divisor,
            batch_size, self.input_size, self.output_size
        )