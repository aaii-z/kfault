import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused linear + reduction operations
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void linear_reduce_kernel(
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

    extern __shared__ float shared_mem[];
    float* row_cache = shared_mem;
    
    // Cache input row
    for (int col = threadIdx.x; col < in_features; col += blockDim.x) {
        row_cache[col] = input[row * in_features + col];
    }
    __syncthreads();

    // Compute linear transformation with bias
    float sum = 0.0f;
    for (int j = 0; j < out_features; ++j) {
        float dot = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            dot += row_cache[k] * weight[j * in_features + k];
        }
        sum += dot + bias[j];
    }

    // Apply reduction operations
    float max_val = sum;
    float mean_val = sum;
    float lse1 = logf(expf(sum));
    float lse2 = logf(expf(lse1));
    
    output[row] = lse2;
}

torch::Tensor linear_reduce_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    auto output = torch::empty({batch_size, 1}, input.options());
    
    dim3 blocks((batch_size + 255) / 256);
    dim3 threads(256);
    size_t shared_mem_size = input.size(1) * sizeof(float);
    
    linear_reduce_kernel<<<blocks, threads, shared_mem_size>>>(
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

cpp_source = "torch::Tensor linear_reduce_cuda(torch::Tensor, torch::Tensor, torch::Tensor);"

# Load the custom CUDA extension
linear_reduce = load_inline(
    name='linear_reduce',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['linear_reduce_cuda'],
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
        return linear_reduce.linear_reduce_cuda(x, self.weight, self.bias)