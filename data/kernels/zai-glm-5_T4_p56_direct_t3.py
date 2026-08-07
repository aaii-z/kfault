import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear-sigmoid-sum
fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for the forward pass: Linear -> Sigmoid -> Sum
// We parallelize over the output elements of the linear layer.
// Each thread computes one element of the linear output, applies sigmoid,
// and atomically adds the result to the corresponding batch sum.
__global__ void fused_linear_sigmoid_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size) {

    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of output elements (batch_size * hidden_size)
    int total_elements = batch_size * hidden_size;

    if (idx < total_elements) {
        // Decompose idx into batch index and hidden index
        int b = idx / hidden_size;
        int h = idx % hidden_size;

        // 1. Compute Linear Layer: y = x * W^T + b
        // We compute the dot product for the specific output neuron 'h' for batch item 'b'.
        float val = bias[h];
        
        // Pointers to the relevant row of input and weight
        const float* input_row = input + b * input_size;
        const float* weight_row = weight + h * input_size;

        // Compute dot product
        for (int i = 0; i < input_size; ++i) {
            val += input_row[i] * weight_row[i];
        }

        // 2. Apply Sigmoid: s = 1 / (1 + exp(-y))
        float sigmoid_val = 1.0f / (1.0f + expf(-val));

        // 3. Sum Reduction
        // Atomically add the result to the output for this batch item.
        atomicAdd(&output[b], sigmoid_val);
    }
}

torch::Tensor fused_linear_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Dimensions
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0); // Weight is (hidden_size, input_size)

    // Output tensor initialized to zero
    // Shape: (batch_size, 1)
    auto output = torch::zeros({batch_size, 1}, input.options());

    // Launch configuration
    // We launch a 1D grid where each thread handles one (batch, hidden) element.
    const int total_elements = batch_size * hidden_size;
    const int block_size = 256; 
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch kernel
    fused_linear_sigmoid_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return output;
}
"""

fused_linear_sigmoid_sum_cpp_source = (
    "torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_linear_sigmoid_sum",
    cpp_sources=fused_linear_sigmoid_sum_cpp_source,
    cuda_sources=fused_linear_sigmoid_sum_source,
    functions=["fused_linear_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused Linear-Sigmoid-Sum.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        # Initialize parameters exactly like nn.Linear to ensure correctness
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        
        # Perform kaiming uniform initialization matching nn.Linear defaults
        nn.init.kaiming_uniform_(self.weight, a=5.0) # sqrt(5) is the default gain
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # Call the custom fused kernel
        return self.fused_op.fused_linear_sigmoid_sum_cuda(x, self.weight, self.bias)