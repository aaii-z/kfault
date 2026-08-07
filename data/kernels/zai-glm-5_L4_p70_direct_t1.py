import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Sigmoid, Scaling, and Residual Add
# This kernel fuses the operations: sigmoid(x) * scaling + x
# It is designed to work on the output of a GEMM operation.
fused_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Block size for the CUDA kernel
const int BLOCK_SIZE = 256;

// Fused kernel: applies sigmoid, scaling, and residual addition element-wise
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

// Helper function to launch the fused kernel
void launch_fused_kernel(const float* input, float* out, int size, float scaling_factor, cudaStream_t stream) {
    const int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_sigmoid_scale_add_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, out, size, scaling_factor);
}

// CUDA wrapper function
torch::Tensor fused_sigmoid_scale_add_cuda(torch::Tensor input, float scaling_factor) {
    // We operate in-place to save memory bandwidth if input is contiguous.
    // However, to be safe and handle strides correctly without complex logic in the kernel,
    // we ensure the output is a contiguous tensor.
    // If input is already contiguous, we can operate in-place.
    
    auto out = torch::empty_like(input);
    
    int size = input.numel();
    
    // Get the current CUDA stream to ensure synchronization
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    launch_fused_kernel(input.data_ptr<float>(), out.data_ptr<float>(), size, scaling_factor, stream);
    
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
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        # We use the standard nn.Linear which is highly optimized (cuBLAS) for the GEMM part.
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        
        # Store the custom op
        self.fused_op = fused_sigmoid_op

    def forward(self, x):
        """
        Forward pass using the custom fused CUDA kernel for the activation part.
        """
        # 1. Perform GEMM (Linear)
        # This uses PyTorch's highly optimized cuBLAS kernel
        x = self.gemm(x)
        
        # 2. Perform Fused Sigmoid, Scaling, and Residual Add
        # This uses our custom CUDA kernel
        x = self.fused_op.fused_sigmoid_scale_add_cuda(x, self.scaling_factor)
        
        return x