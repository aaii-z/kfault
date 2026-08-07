import torch
import torch.nn as nn
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
    // Grid is launched as (batch_size, hidden_size)
    int b = blockIdx.x; // Batch index
    int h = threadIdx.x; // Hidden dimension index (assuming blockDim.x >= hidden_size)

    if (h < hidden_size) {
        // 1. Compute Linear Layer: y = x * W^T + b
        // We compute the dot product for the specific output neuron 'h' for batch item 'b'.
        float val = bias[h];
        
        // Optimized memory access: input is row-major, weight is row-major (output features x input features)
        // We iterate through the input dimension.
        // Using restrict and const pointers helps the compiler optimize memory loads.
        const float* input_row = input + b * input_size;
        const float* weight_row = weight + h * input_size;

        // Loop unrolling or vectorization could be added here for further optimization,
        // but for simplicity and general case, a straight loop is used.
        // Modern CUDA compilers do a decent job with simple loops.
        for (int i = 0; i < input_size; ++i) {
            val += input_row[i] * weight_row[i];
        }

        // 2. Apply Sigmoid: s = 1 / (1 + exp(-y))
        float sigmoid_val = 1.0f / (1.0f + expf(-val));

        // 3. Sum Reduction
        // We need to sum sigmoid_val into output[b].
        // Since threads from the same block access the same output address (output[b]),
        // we can use atomicAdd.
        // Note: For hidden_size = 32768, we have 32768 threads atomically adding to the same address.
        // This causes high contention. However, given the problem constraints and the goal of fusion,
        // this is the direct approach. A more optimized reduction would use shared memory,
        // but adds significant complexity to the kernel logic.
        // Given the large hidden_size, atomicAdd is a practical fused solution.
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
    // We launch a grid where each block corresponds to a batch item.
    // Block size must be large enough to cover hidden_size.
    // Max threads per block is typically 1024. hidden_size is 32768.
    // This means we cannot launch a single block per row if we map one thread per hidden element.
    // We need to map threads differently or use a grid-stride loop.
    // Let's use a 1D grid mapping where each thread handles one (batch, hidden) element.
    
    // Correct Kernel Launch Strategy:
    // Grid: (batch_size * hidden_size) threads in total.
    // Or better: Grid (batch_size) and Block (hidden_size) is not possible if hidden > 1024.
    // We will use a 1D grid of size (batch_size * hidden_size).
    
    const int total_elements = batch_size * hidden_size;
    const int block_size = 256; // Standard block size
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // We need to pass dimensions explicitly to the kernel
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
        # We keep the parameters as standard Parameters to maintain compatibility
        # with PyTorch's initialization and optimization utilities.
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
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