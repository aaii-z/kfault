import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized forward pass
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void optimized_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;

    float sum_val = 0.0f;
    for (int col = 0; col < out_features; ++col) {
        float dot_product = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            dot_product += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum_val += dot_product + bias[col];
    }

    // Apply reduction operations (all are identity operations on scalar values)
    float max_val = sum_val;
    float mean_val = max_val;
    float lse1 = logf(expf(mean_val));
    float lse2 = logf(expf(lse1));
    
    output[row] = lse2;
}

torch::Tensor optimized_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    auto output = torch::empty({batch_size, 1}, input.options());
    
    dim3 blocks((batch_size + 255) / 256);
    dim3 threads(256);
    
    optimized_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input.size(1),
        weight.size(0)
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor optimized_forward_cuda(torch::Tensor, torch::Tensor, torch::Tensor);"

# Load the custom CUDA extension
optimized_forward = load_inline(
    name='optimized_forward',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['optimized_forward_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return optimized_forward.optimized_forward_cuda(x, self.weight, self.bias)