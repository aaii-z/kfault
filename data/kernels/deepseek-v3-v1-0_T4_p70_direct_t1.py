import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM + Sigmoid + Scaling + Residual Add CUDA kernel
fused_gemm_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void sigmoid_scaling_residual_kernel(float* output, const float* gemm_result, 
                                               float scaling_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_val = 1.0f / (1.0f + expf(-gemm_result[idx]));
        output[idx] = sigmoid_val * scaling_factor + gemm_result[idx];
    }
}

torch::Tensor fused_gemm_sigmoid_scaling_residual_cuda(torch::Tensor input, torch::Tensor weight, 
                                                      torch::Tensor bias, float scaling_factor) {
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight.size(0);
    
    // Allocate output tensor
    auto output = torch::empty({batch_size, hidden_size}, input.options());
    auto gemm_result = torch::empty({batch_size, hidden_size}, input.options());
    
    // Perform GEMM using cuBLAS
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // input * weight^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                hidden_size, batch_size, input_size, 
                &alpha, 
                weight.data_ptr<float>(), hidden_size,
                input.data_ptr<float>(), input_size,
                &beta,
                gemm_result.data_ptr<float>(), hidden_size);
    
    // Add bias
    if (bias.defined()) {
        const float bias_alpha = 1.0f;
        const float bias_beta = 1.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    hidden_size, batch_size, 1,
                    &bias_alpha,
                    bias.data_ptr<float>(), hidden_size,
                    torch::ones({1, batch_size}, input.options()).data_ptr<float>(), 1,
                    &bias_beta,
                    gemm_result.data_ptr<float>(), hidden_size);
    }
    
    // Apply sigmoid, scaling, and residual addition in a single kernel
    int total_elements = batch_size * hidden_size;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    sigmoid_scaling_residual_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), gemm_result.data_ptr<float>(),
        scaling_factor, total_elements
    );
    
    return output;
}
"""

fused_gemm_activation_cpp_source = (
    "torch::Tensor fused_gemm_sigmoid_scaling_residual_cuda(torch::Tensor input, torch::Tensor weight, "
    "torch::Tensor bias, float scaling_factor);"
)

# Compile the inline CUDA code
fused_gemm_activation = load_inline(
    name="fused_gemm_activation",
    cpp_sources=fused_gemm_activation_cpp_source,
    cuda_sources=fused_gemm_activation_source,
    functions=["fused_gemm_sigmoid_scaling_residual_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd" with fused CUDA operations.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scaling_factor = scaling_factor
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=5.0**0.5)
        nn.init.zeros_(self.bias)
        
        self.fused_op = fused_gemm_activation

    def forward(self, x):
        """
        Forward pass of the optimized model using fused CUDA operations.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        return self.fused_op.fused_gemm_sigmoid_scaling_residual_cuda(
            x, self.weight, self.bias, self.scaling_factor
        )