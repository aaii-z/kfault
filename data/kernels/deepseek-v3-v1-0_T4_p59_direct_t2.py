import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + swish + scaling
fused_swish_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float swish_activation(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

__global__ void fused_swish_matmul_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output, 
    int batch_size, 
    int in_features, 
    int out_features,
    float scaling_factor) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        if (bias) {
            sum = bias[col];
        }
        
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        // Apply Swish activation and scaling
        float activated = sum * (1.0f / (1.0f + expf(-sum)));
        output[row * out_features + col] = activated * scaling_factor;
    }
}

torch::Tensor fused_swish_matmul_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    float scaling_factor) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );
    
    fused_swish_matmul_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor
    );
    
    return output;
}
"""

fused_swish_matmul_cpp_source = """
torch::Tensor fused_swish_matmul_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);
"""

# Compile the inline CUDA code
fused_swish_matmul = load_inline(
    name="fused_swish_matmul",
    cpp_sources=fused_swish_matmul_cpp_source,
    cuda_sources=fused_swish_matmul_source,
    functions=["fused_swish_matmul_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.fused_op = fused_swish_matmul
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.fused_op.fused_swish_matmul_cuda(x, self.weight, self.bias, self.scaling_factor)