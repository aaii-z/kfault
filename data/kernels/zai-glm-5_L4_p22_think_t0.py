import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__device__ float mish_activation(float x) {
    // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    float sp = log1pf(expf(x));
    return x * tanhf(sp);
}

__global__ void fused_ops_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                 int hidden_size, float scale, float clamp_min, float clamp_max) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* row_ptr = input + row * hidden_size;
    
    // Online softmax reduction variables
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    // Effective scale: scale_factor * 2 (because x = x + x)
    float effective_scale = scale * 2.0f;
    
    // Stride loop to process the row
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = row_ptr[i];
        // Apply scale
        val = val * effective_scale;
        // Apply clamp
        val = fmaxf(clamp_min, fminf(clamp_max, val));
        
        // Online softmax update
        float new_max = fmaxf(max_val, val);
        float correction = expf(max_val - new_max);
        sum_exp = sum_exp * correction + expf(val - new_max);
        max_val = new_max;
    }
    
    // Shared memory for block reduction
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];
    
    s_max[tid] = max_val;
    s_sum[tid] = sum_exp;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float my_max = s_max[tid];
            float my_sum = s_sum[tid];
            float other_max = s_max[tid + s];
            float other_sum = s_sum[tid + s];
            
            float new_max = fmaxf(my_max, other_max);
            float new_sum = my_sum * expf(my_max - new_max) + other_sum * expf(other_max - new_max);
            
            s_max[tid] = new_max;
            s_sum[tid] = new_sum;
        }
        __syncthreads();
    }
    
    // Thread 0 computes the final result
    if (tid == 0) {
        float lse = s_max[0] + logf(s_sum[0]);
        // Apply Mish: x * mish(x)
        float mish_val = mish_activation(lse);
        output[row] = lse * mish_val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, float scale, float clamp_min, float clamp_max) {
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    auto output = torch::empty({batch_size, 1}, input.options());
    
    fused_ops_kernel<<<batch_size, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        hidden_size,
        scale,
        clamp_min,
        clamp_max
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, float scale, float clamp_min, float clamp_max);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication followed by a fused CUDA kernel.
    The fused kernel combines scaling, residual addition, clamping, LogSumExp, and Mish activation.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # Step 1: Matrix Multiplication (using optimized PyTorch Linear)
        x = self.matmul(x)
        
        # Step 2: Fused CUDA kernel for Scale, Add, Clamp, LogSumExp, and Mish
        x = self.fused_ops.fused_ops_cuda(x, self.scale_factor, self.clamp_min, self.clamp_max)
        
        return x