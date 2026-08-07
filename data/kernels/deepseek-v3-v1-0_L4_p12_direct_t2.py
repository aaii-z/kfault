import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM-Multiply-LeakyReLU CUDA kernel
fused_gemm_multiply_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void multiply_leakyrelu_kernel(float* output, float multiplier, float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = output[idx] * multiplier;
        output[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor fused_gemm_multiply_leakyrelu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float multiplier, 
    float negative_slope) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Use cuBLAS for efficient GEMM
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Perform matrix multiplication: output = input * weight^T
    // Note: cuBLAS expects column-major, so we need to transpose the operation
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_T,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), out_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    // Add bias to each row
    if (bias.defined()) {
        for (int i = 0; i < batch_size; ++i) {
            float alpha_bias = 1.0f;
            cublasSaxpy(handle, out_features, &alpha_bias, 
                        bias.data_ptr<float>(), 1,
                        output.data_ptr<float>() + i * out_features, 1);
        }
    }
    
    cublasDestroy(handle);
    
    // Apply element-wise multiply and LeakyReLU
    int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    multiply_leakyrelu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), multiplier, negative_slope, total_elements);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_gemm_multiply_leakyrelu_cpp_source = (
    "torch::Tensor fused_gemm_multiply_leakyrelu_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "float multiplier, float negative_slope);"
)

# Compile the inline CUDA code
fused_gemm_multiply_leakyrelu = load_inline(
    name="fused_gemm_multiply_leakyrelu",
    cpp_sources=fused_gemm_multiply_leakyrelu_cpp_source,
    cuda_sources=fused_gemm_multiply_leakyrelu_source,
    functions=["fused_gemm_multiply_leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=["-lcublas"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        self.fused_op = fused_gemm_multiply_leakyrelu

    def forward(self, x):
        return self.fused_op.fused_gemm_multiply_leakyrelu_cuda(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )