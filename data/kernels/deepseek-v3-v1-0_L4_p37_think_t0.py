import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused Swish+Bias CUDA kernel
swish_bias_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void swish_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int row = idx / out_features;
        int col = idx % out_features;
        float x = input[idx] + bias[col];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid_x;
    }
}

torch::Tensor swish_bias_cuda(torch::Tensor input, torch::Tensor bias) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    auto output = torch::empty_like(input);
    
    int threads = 256;
    int blocks = (batch_size * out_features + threads - 1) / threads;
    
    swish_bias_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return output;
}
"""

swish_bias_cpp_source = "torch::Tensor swish_bias_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the custom CUDA operation
swish_bias_op = load_inline(
    name="swish_bias",
    cpp_sources=swish_bias_cpp_source,
    cuda_sources=swish_bias_source,
    functions=["swish_bias_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.swish_bias = swish_bias_op

    def forward(self, x):
        x = self.matmul(x)
        x = self.swish_bias.swish_bias_cuda(x, self.bias)
        x = self.group_norm(x)
        return x