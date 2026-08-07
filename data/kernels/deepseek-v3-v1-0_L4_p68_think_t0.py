import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + min + subtract operation
fused_linear_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_linear_min_sub_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float constant,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float value = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            value += input[row * in_features + k] * weight[col * in_features + k];
        }
        value += bias[col];
        value = min(value, constant);
        value -= constant;
        output[row * out_features + col] = value;
    }
}

torch::Tensor fused_linear_min_sub_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float constant
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);

    fused_linear_min_sub_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        constant,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_linear_cpp_source = """
torch::Tensor fused_linear_min_sub_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float constant);
"""

# Compile the inline CUDA code
fused_linear_op = load_inline(
    name="fused_linear_op",
    cpp_sources=fused_linear_cpp_source,
    cuda_sources=fused_linear_kernel_source,
    functions=["fused_linear_min_sub_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.constant = nn.Parameter(torch.tensor(constant), requires_grad=False)
        
        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return fused_linear_op.fused_linear_min_sub_cuda(x, self.weight, self.bias, self.constant.item())