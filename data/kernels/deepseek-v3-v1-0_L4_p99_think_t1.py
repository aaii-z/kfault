import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for fused Linear+GELU
fused_linear_gelu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void linear_gelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int in_features,
    int out_features,
    int batch_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum += bias[col];
        output[row * out_features + col] = gelu(sum);
    }
}

torch::Tensor linear_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 block(16, 16);
    dim3 grid((batch_size + block.x - 1) / block.x, (out_features + block.y - 1) / block.y);

    linear_gelu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        in_features,
        out_features,
        batch_size
    );

    return output;
}
"""

fused_linear_gelu_cpp_source = """
torch::Tensor linear_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_linear_gelu = load_inline(
    name="fused_linear_gelu",
    cpp_sources=fused_linear_gelu_cpp_source,
    cuda_sources=fused_linear_gelu_source,
    functions=["linear_gelu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = fused_linear_gelu.linear_gelu_cuda(x, self.weight, self.bias)
        x = torch.nn.functional.softmax(x, dim=1)
        return x