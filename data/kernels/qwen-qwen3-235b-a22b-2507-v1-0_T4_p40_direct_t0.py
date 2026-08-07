import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul, scale, and residual addition
fused_matmul_scale_residual_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_matmul_scale_residual_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float scaling_factor,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int row = idx / out_features;
    int col = idx % out_features;

    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];  // Add bias

    // Apply scaling and residual: output = sum * scaling_factor + sum
    output[idx] = sum * scaling_factor + sum;
}

torch::Tensor fused_matmul_scale_residual_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = bias.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_matmul_scale_residual_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_matmul_scale_residual_cpp_source = """
torch::Tensor fused_matmul_scale_residual_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_matmul_scale_residual",
    cpp_sources=fused_matmul_scale_residual_cpp_source,
    cuda_sources=fused_matmul_scale_residual_source,
    functions=["fused_matmul_scale_residual_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matrix multiplication, scaling, and residual addition.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_op.fused_matmul_scale_residual_cuda(x, self.weight, self.bias, self.scaling_factor)