import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel for matmul, subtract, multiply, and ReLU
fused_operation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operation_kernel(
    float* output, 
    const float* input, 
    const float* weight, 
    const float* bias,
    float subtract_value, 
    float multiply_value, 
    int batch_size, 
    int in_features, 
    int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        sum = (sum - subtract_value) * multiply_value;
        sum = fmaxf(0.0f, sum);
        
        output[row * out_features + col] = sum;
    }
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    float subtract_value, 
    float multiply_value) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );
    
    fused_operation_kernel<<<grid_size, block_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        subtract_value,
        multiply_value,
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_operation_cpp_source = """
torch::Tensor fused_operation_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    float subtract_value, 
    float multiply_value);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_operation_cpp_source,
    cuda_sources=fused_operation_source,
    functions=["fused_operation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        
        # Initialize linear layer parameters manually
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_operation = fused_operation

    def forward(self, x):
        return self.fused_operation.fused_operation_cuda(
            x, self.weight, self.bias, self.subtract_value, self.multiply_value
        )