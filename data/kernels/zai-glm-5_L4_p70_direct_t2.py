import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Sigmoid, Scaling, and Residual Add
fused_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_sigmoid_scale_add_kernel(const float* __restrict__ input, float* __restrict__ out, int size, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Sigmoid function: 1 / (1 + exp(-x))
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        // Scale and add residual
        out[idx] = sigmoid_val * scaling_factor + val;
    }
}

torch::Tensor fused_sigmoid_scale_add_cuda(torch::Tensor input, float scaling_factor) {
    auto out = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_sigmoid_scale_add_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), out.data_ptr<float>(), size, scaling_factor);
    
    return out;
}
"""

fused_sigmoid_cpp_source = (
    "torch::Tensor fused_sigmoid_scale_add_cuda(torch::Tensor input, float scaling_factor);"
)

# Compile the inline CUDA code
fused_sigmoid_op = load_inline(
    name="fused_sigmoid",
    cpp_sources=fused_sigmoid_cpp_source,
    cuda_sources=fused_sigmoid_source,
    functions=["fused_sigmoid_scale_add_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_sigmoid_op

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_sigmoid_scale_add_cuda(x, self.scaling_factor)
        return x