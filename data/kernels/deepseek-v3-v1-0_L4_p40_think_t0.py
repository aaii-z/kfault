import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear, scaling, and residual addition
fused_linear_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_linear_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float scaling_factor,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        if (bias) {
            sum += bias[col];
        }
        float original = sum;
        sum = sum * scaling_factor + original;
        output[row * out_features + col] = sum;
    }
}

torch::Tensor fused_linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 blocks(
        (out_features + 15) / 16,
        (batch_size + 15) / 16
    );
    dim3 threads(16, 16);

    fused_linear_kernel<<<blocks, threads>>>(
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

fused_linear_cpp_source = "torch::Tensor fused_linear_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);"

# Compile the inline CUDA code
fused_linear = load_inline(
    name="fused_linear",
    cpp_sources=fused_linear_cpp_source,
    cuda_sources=fused_linear_source,
    functions=["fused_linear_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_linear.fused_linear_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.scaling_factor
        )