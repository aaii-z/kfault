import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel for element-wise multiply and LeakyReLU
fused_multiply_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_multiply_leakyrelu_kernel(
    const float* input,
    float* output,
    int numel,
    float multiplier,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = input[idx] * multiplier;
        output[idx] = val >= 0.0f ? val : val * negative_slope;
    }
}

torch::Tensor fused_multiply_leakyrelu_cuda(
    torch::Tensor input,
    float multiplier,
    float negative_slope
) {
    auto output = torch::empty_like(input);
    int numel = input.numel();
    int block_size = 256;
    int num_blocks = (numel + block_size - 1) / block_size;
    
    fused_multiply_leakyrelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        multiplier,
        negative_slope
    );
    
    return output;
}
"""

fused_multiply_leakyrelu_cpp_source = """
torch::Tensor fused_multiply_leakyrelu_cuda(torch::Tensor input, float multiplier, float negative_slope);
"""

# Load the custom CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=fused_multiply_leakyrelu_cpp_source,
    cuda_sources=fused_multiply_leakyrelu_source,
    functions=['fused_multiply_leakyrelu_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.fused_multiply_leakyrelu = fused_ops.fused_multiply_leakyrelu_cuda

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_multiply_leakyrelu(x, self.multiplier, self.negative_slope)
        return x