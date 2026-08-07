import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + mish + mish
fused_linear_mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr float MISH_THRESHOLD = 20.0f;

__device__ float mish_activation(float x) {
    if (x > MISH_THRESHOLD) {
        return x;
    } else if (x < -MISH_THRESHOLD) {
        return 0.0f * x; // ensures proper gradient
    }
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_linear_mish_mish_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, 
    int in_features, 
    int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        
        // Matrix multiplication
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        // Add bias
        if (bias) {
            sum += bias[col];
        }
        
        // Apply Mish activation twice
        float result = mish_activation(sum);
        result = mish_activation(result);
        
        output[row * out_features + col] = result;
    }
}

torch::Tensor fused_linear_mish_mish_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );
    
    fused_linear_mish_mish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_linear_mish_mish_cpp_source = "torch::Tensor fused_linear_mish_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
fused_linear_mish_mish = load_inline(
    name="fused_linear_mish_mish",
    cpp_sources=fused_linear_mish_mish_cpp_source,
    cuda_sources=fused_linear_mish_mish_source,
    functions=["fused_linear_mish_mish_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias like nn.Linear
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5.0**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_linear_mish_mish.fused_linear_mish_mish_cuda(x, self.weight, self.bias)