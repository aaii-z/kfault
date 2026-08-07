import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused bias + ReLU CUDA kernel
fused_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_relu_kernel(float* output, const float* bias, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    if (idx < total_elements) {
        int col = idx % out_features;
        float value = output[idx] + bias[col];
        output[idx] = value > 0 ? value : 0.0f;
    }
}

torch::Tensor fused_bias_relu_cuda(torch::Tensor output, torch::Tensor bias) {
    auto batch_size = output.size(0);
    auto out_features = output.size(1);
    int total_elements = batch_size * out_features;
    
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    bias_relu_kernel<<<num_blocks, threads_per_block>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return output;
}
"""

fused_bias_relu_cpp_source = "torch::Tensor fused_bias_relu_cuda(torch::Tensor output, torch::Tensor bias);"

# Compile the inline CUDA code
fused_bias_relu = load_inline(
    name="fused_bias_relu",
    cpp_sources=fused_bias_relu_cpp_source,
    cuda_sources=fused_bias_relu_source,
    functions=["fused_bias_relu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_bias_relu = fused_bias_relu

    def forward(self, x):
        x = torch.matmul(x, self.weight.t())
        x = self.fused_bias_relu.fused_bias_relu_cuda(x, self.bias)
        return x