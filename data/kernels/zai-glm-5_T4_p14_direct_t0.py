import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ out, int batch_size, int input_size, int hidden_size, float scaling_factor) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (batch_idx >= batch_size) return;

    // Pointers for the current batch element
    const float* x_row = x + batch_idx * input_size;
    
    // Shared memory for reduction
    // We need to reduce across hidden_size. 
    // We compute partial sums in shared memory.
    extern __shared__ float s_partial_sums[];

    float thread_sum = 0.0f;

    // Each thread computes a partial sum over the hidden dimension
    // Loop over hidden_size with a stride of block_size
    for (int h_idx = tid; h_idx < hidden_size; h_idx += block_size) {
        float dot_product = 0.0f;
        const float* weight_row = weight + h_idx * input_size;
        
        // Compute dot product: x[batch] dot weight[h_idx]
        for (int k = 0; k < input_size; ++k) {
            dot_product += x_row[k] * weight_row[k];
        }
        
        // Apply division by 2 and accumulate
        thread_sum += dot_product * 0.5f;
    }

    s_partial_sums[tid] = thread_sum;
    __syncthreads();

    // Block reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partial_sums[tid] += s_partial_sums[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        out[batch_idx] = s_partial_sums[0] * scaling_factor;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor) {
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int hidden_size = weight.size(0);

    auto out = torch::empty({batch_size, 1}, x.options());

    const int block_size = 256;
    const int num_blocks = batch_size;
    // Calculate shared memory size
    size_t shared_mem_size = block_size * sizeof(float);

    fused_op_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        scaling_factor
    );

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor);"
)

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs the matrix multiplication, division, summation, and scaling
    in a single fused CUDA kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.fused_op.fused_op_cuda(x, self.weight, self.scaling_factor)