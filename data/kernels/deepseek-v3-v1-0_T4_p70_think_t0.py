import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Sigmoid-Scaling-Add operation
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_sigmoid_scaling_add_kernel(
    const float* input,
    float* output,
    float scaling_factor,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        output[idx] = sigmoid_val * scaling_factor + val;
    }
}

torch::Tensor fused_sigmoid_scaling_add_cuda(
    torch::Tensor input,
    float scaling_factor
) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    fused_sigmoid_scaling_add_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        num_elements
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_sigmoid_scaling_add_cuda(torch::Tensor input, float scaling_factor);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_sigmoid_scaling_add_cuda"],
    verbose=True,
    extra_cflags=[],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model implementing fused "Sigmoid-Scaling-ResidualAdd" operation.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        """
        Forward pass with fused operations.
        """
        x = self.gemm(x)
        x = self.fused_op.fused_sigmoid_scaling_add_cuda(x, self.scaling_factor)
        return x