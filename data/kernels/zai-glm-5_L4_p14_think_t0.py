import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMV and scaling
# This kernel replaces the sequence: MatMul -> Div -> Sum -> Scale
# Algorithmic optimization:
# Original: y = x @ W.T -> y / 2 -> sum(y, dim=1) -> y * scale
# Optimized: y = scale/2 * (x @ sum(W, dim=0))
# This reduces the complexity from O(B * I * H) to O(B * I + I * H) by pre-reducing weights.

gemv_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemv_scale_kernel(const float* __restrict__ matrix, const float* __restrict__ vector, float* __restrict__ out, float scale, int batch_size, int input_size) {
    // Dynamic shared memory for the vector and reduction buffer
    // Layout: [input_size floats for vector][blockDim.x floats for reduction]
    extern __shared__ float s_buffer[];
    float* s_vector = s_buffer;
    float* s_reduce = s_buffer + input_size;

    int row_idx = blockIdx.x;
    if (row_idx >= batch_size) return;

    // 1. Load vector into shared memory
    // Each thread loads multiple elements to utilize memory bandwidth
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
        s_vector[i] = vector[i];
    }
    __syncthreads();

    // 2. Compute dot product of matrix row and vector
    // matrix is (batch_size, input_size), vector is (input_size,)
    float sum = 0.0f;
    const float* row_ptr = matrix + row_idx * input_size;
    
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
        sum += row_ptr[i] * s_vector[i];
    }

    // 3. Parallel reduction within the block
    s_reduce[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
        }
        __syncthreads();
    }

    // 4. Write result
    if (threadIdx.x == 0) {
        out[row_idx] = s_reduce[0] * scale;
    }
}

torch::Tensor gemv_scale_cuda(torch::Tensor matrix, torch::Tensor vector, float scale) {
    int batch_size = matrix.size(0);
    int input_size = matrix.size(1);
    
    auto out = torch::zeros({batch_size, 1}, matrix.options());

    const int block_size = 256;
    const int num_blocks = batch_size;
    
    // Shared memory size: input_size for vector + block_size for reduction
    size_t shared_mem_size = (input_size + block_size) * sizeof(float);

    gemv_scale_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        matrix.data_ptr<float>(), 
        vector.data_ptr<float>(), 
        out.data_ptr<float>(), 
        scale, 
        batch_size, 
        input_size
    );

    return out;
}
"""

gemv_scale_cpp_source = (
    "torch::Tensor gemv_scale_cuda(torch::Tensor matrix, torch::Tensor vector, float scale);"
)

# Compile the inline CUDA code
gemv_scale_op = load_inline(
    name="gemv_scale",
    cpp_sources=gemv_scale_cpp_source,
    cuda_sources=gemv_scale_source,
    functions=["gemv_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication, division, summation, and scaling.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.gemv_scale = gemv_scale_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Algorithmic optimization: Reduce weights along the hidden dimension
        # The original operation sum(x @ W.T, dim=1) is equivalent to x @ sum(W, dim=0)
        # This transforms a (B, I) x (H, I).T -> (B, H) sum -> (B, 1) operation
        # into a (B, I) x (I,) -> (B, 1) operation.
        # We compute the reduced weight vector once.
        w_sum = self.weight.sum(dim=0)
        
        # The scaling factor combines the division by 2 and the multiplication by scaling_factor
        combined_scale = self.scaling_factor * 0.5
        
        # Use the custom CUDA kernel for fused GEMV and scaling
        return self.gemv_scale.gemv_scale_cuda(x, w_sum, combined_scale)