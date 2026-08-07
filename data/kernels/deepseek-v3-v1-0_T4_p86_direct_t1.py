import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel for linear + division + GELU
fused_linear_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr float GELU_COEF_A = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__global__ void fused_linear_gelu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float divisor, int batch_size, int input_size, int output_size) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < output_size) {
        float sum = 0.0f;
        for (int k = 0; k < input_size; ++k) {
            sum += input[row * input_size + k] * weight[col * input_size + k];
        }
        
        if (bias) {
            sum += bias[col];
        }
        
        sum /= divisor;
        output[row * output_size + col] = gelu_activation(sum);
    }
}

torch::Tensor fused_linear_gelu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor) {
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(0);
    
    auto output = torch::empty({batch_size, output_size}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (output_size + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );
    
    fused_linear_gelu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        divisor,
        batch_size,
        input_size,
        output_size
    );
    
    return output;
}
"""

fused_linear_gelu_cpp_source = "torch::Tensor fused_linear_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);"

# Compile the inline CUDA code
fused_linear_gelu = load_inline(
    name="fused_linear_gelu",
    cpp_sources=fused_linear_gelu_cpp_source,
    cuda_sources=fused_linear_gelu_source,
    functions=["fused_linear_gelu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.divisor = divisor
        
        # Initialize weights and bias like nn.Linear
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_linear_gelu.fused_linear_gelu_cuda(x, self.weight, self.bias, self.divisor)