import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias addition, division, and Swish activation
fused_bias_div_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_bias_div_swish_kernel(
    const float* input,
    const float* bias,
    float divide_value,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp = (input[idx] + bias[0]) / divide_value;
        output[idx] = temp * sigmoid(temp);
    }
}

torch::Tensor fused_bias_div_swish_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float divide_value
) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    
    fused_bias_div_swish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        divide_value,
        output.data_ptr<float>(),
        n
    );
    
    return output;
}
"""

fused_bias_div_swish_cpp_source = "torch::Tensor fused_bias_div_swish_cuda(torch::Tensor input, torch::Tensor bias, float divide_value);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=fused_bias_div_swish_cpp_source,
    cuda_sources=fused_bias_div_swish_source,
    functions=['fused_bias_div_swish_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x):
        x = self.matmul(x)
        x = self.bn(x)
        x = fused_ops.fused_bias_div_swish_cuda(x, self.bias, self.divide_value)
        return x