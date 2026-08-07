import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, Sigmoid, Scaling, and Residual Add
# This kernel fuses the operations: sigmoid(x) * scaling + x
# where x is the output of the GEMM operation.
fused_gemm_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Block size for the CUDA kernel
const int BLOCK_SIZE = 256;

// Fused kernel: applies sigmoid, scaling, and residual addition element-wise
__global__ void fused_sigmoid_scale_add_kernel(const float* __restrict__ gemm_out, float* __restrict__ out, int size, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = gemm_out[idx];
        // Sigmoid function: 1 / (1 + exp(-x))
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        // Scale and add residual
        out[idx] = sigmoid_val * scaling_factor + val;
    }
}

// Helper function to launch the fused kernel
void launch_fused_kernel(const float* gemm_out, float* out, int size, float scaling_factor, cudaStream_t stream) {
    const int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_sigmoid_scale_add_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(gemm_out, out, size, scaling_factor);
}

// CUDA wrapper function
torch::Tensor fused_gemm_sigmoid_scale_add_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float scaling_factor) {
    // Perform GEMM using cuBLAS (handled by PyTorch's matmul)
    // We compute x @ weight.T + bias
    // Note: PyTorch's Linear stores weights as (out_features, in_features), so we transpose it for matmul
    // However, using torch::mm handles the layout correctly if we use the appropriate functions.
    // torch::linear is available and handles the bias addition efficiently.
    
    // We perform the GEMM into a temporary buffer or the output buffer directly.
    // Since we need the original GEMM output for the residual addition, we must store it.
    // We can compute the GEMM result into the output tensor 'out', then apply the fused operation in-place or to a new buffer.
    // To save memory bandwidth, we compute GEMM into 'out', then read 'out', compute activation, and write back to 'out'.
    // This is valid because the write happens after the read for a given element if we launch a new kernel.
    // However, to ensure correctness with separate kernels, we need to synchronize or use streams.
    // A better approach for a single kernel would require writing a custom GEMM, which is complex and likely slower than cuBLAS.
    // The strategy here is:
    // 1. Allocate output tensor.
    // 2. Compute GEMM (x @ W^T + b) into the output tensor using PyTorch's optimized matmul.
    // 3. Launch a custom kernel that reads from the output tensor, applies sigmoid/scale/add, and writes back.
    // This saves the memory bandwidth of reading the GEMM output again in a separate op and allocating an intermediate buffer.
    
    auto out = torch::empty({x.size(0), weight.size(0)}, x.options());
    
    // Use torch::matmul for the GEMM. 
    // PyTorch linear: output = input.matmul(weight.t()) + bias
    // We perform this operation into 'out'.
    // We can use the C++ API for this.
    // Note: torch::mm performs matrix multiplication. torch::addmm performs alpha * mat1 * mat2 + beta * mat3.
    // We want out = x * W^T + bias.
    // We can use at::addmm_out to compute directly into 'out'.
    
    // at::addmm_out(out, bias, x, weight.transpose(0, 1), 1.0, 1.0);
    // However, bias is 1D. addmm expects 2D. 
    // torch::linear handles this.
    
    out = torch::linear(x, weight, bias);
    
    // Now apply the fused kernel in-place on 'out'
    int size = out.numel();
    
    // Get the current CUDA stream to ensure synchronization
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    launch_fused_kernel(out.data_ptr<float>(), out.data_ptr<float>(), size, scaling_factor, stream);
    
    return out;
}
"""

fused_gemm_sigmoid_cpp_source = (
    "torch::Tensor fused_gemm_sigmoid_scale_add_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_gemm_op = load_inline(
    name="fused_gemm_sigmoid",
    cpp_sources=fused_gemm_sigmoid_cpp_source,
    cuda_sources=fused_gemm_sigmoid_source,
    functions=["fused_gemm_sigmoid_scale_add_cuda"],
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
        # We define the parameters manually to pass them to the custom op
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.scaling_factor = scaling_factor
        
        # Initialize parameters similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fused_op = fused_gemm_op

    def forward(self, x):
        """
        Forward pass using the custom fused CUDA kernel.
        """
        return self.fused_op.fused_gemm_sigmoid_scale_add_cuda(x, self.weight, self.bias, self.scaling_factor)

import math