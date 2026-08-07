import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused matmul, divide, sum, and scale operation
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_matmul_divide_sum_scale_kernel(
    const float* input,
    const float* weight,
    float* output,
    float scaling_factor,
    int batch_size,
    int input_size,
    int hidden_size) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size) {
        float sum = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            float val = 0.0f;
            for (int k = 0; k < input_size; ++k) {
                val += input[row * input_size + k] * weight[h * input_size + k];
            }
            sum += val * 0.5f;  // Division by 2 as multiplication by 0.5
        }
        output[row] = sum * scaling_factor;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    float scaling_factor) {
    
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::empty({batch_size}, x.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    fused_matmul_divide_sum_scale_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        input_size,
        hidden_size
    );

    return output.unsqueeze(1); // Reshape to (batch_size, 1)
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul, divide, sum, and scale.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(x, self.weight, self.scaling_factor)