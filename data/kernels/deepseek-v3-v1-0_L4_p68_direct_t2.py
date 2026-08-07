import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for linear + min + subtract
fused_linear_min_sub_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_min_sub_kernel(
    const float* input, const float* weight, const float* bias, 
    float* output, float constant, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        if (bias) {
            sum += bias[col];
        }
        
        // Apply min(constant) and subtract constant
        sum = fminf(sum, constant) - constant;
        
        output[row * out_features + col] = sum;
    }
}

torch::Tensor fused_linear_min_sub_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float constant) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );
    
    fused_linear_min_sub_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        constant,
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_linear_min_sub_cpp_source = """
torch::Tensor fused_linear_min_sub_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float constant);
"""

# Compile the inline CUDA code
fused_linear_min_sub = load_inline(
    name="fused_linear_min_sub",
    cpp_sources=fused_linear_min_sub_cpp_source,
    cuda_sources=fused_linear_min_sub_source,
    functions=["fused_linear_min_sub_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.constant = nn.Parameter(torch.tensor(constant))
        self.fused_op = fused_linear_min_sub

    def forward(self, x):
        return self.fused_op.fused_linear_min_sub_cuda(x, self.weight, self.bias, self.constant.item())