import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear, division, and GELU
fused_linear_div_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation function (approximation)
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Fused kernel: matrix multiplication + division + GELU
__global__ void fused_linear_div_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int output_size,
    float divisor) {
    
    // Calculate output element index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_size;
    
    if (idx < total_elements) {
        int row = idx / output_size;  // batch index
        int col = idx % output_size;  // output feature index
        
        // Compute dot product for this output element
        float sum = 0.0f;
        
        // Use shared memory for weights if beneficial (simplified version without shared memory)
        for (int k = 0; k < input_size; ++k) {
            sum += input[row * input_size + k] * weight[col * input_size + k];
        }
        
        // Add bias
        sum += bias[col];
        
        // Divide by scalar
        sum = sum / divisor;
        
        // Apply GELU activation
        output[idx] = gelu(sum);
    }
}

torch::Tensor fused_linear_div_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor) {
    
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * output_size + block_size - 1) / block_size;
    
    fused_linear_div_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size,
        divisor);
    
    return output;
}
"""

fused_linear_div_gelu_cpp_source = (
    "torch::Tensor fused_linear_div_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);"
)

# Compile the inline CUDA code for fused operations
fused_linear_div_gelu = load_inline(
    name="fused_linear_div_gelu",
    cpp_sources=fused_linear_div_gelu_cpp_source,
    cuda_sources=fused_linear_div_gelu_source,
    functions=["fused_linear_div_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused linear, division, and GELU.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        # Initialize weights and bias like nn.Linear
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.divisor = divisor
        
        # Use the custom CUDA kernel
        self.fused_op = fused_linear_div_gelu

    def forward(self, x):
        return self.fused_op.fused_linear_div_gelu_cuda(x, self.weight, self.bias, self.divisor)