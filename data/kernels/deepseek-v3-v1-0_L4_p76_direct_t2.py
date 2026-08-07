import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM + bias + ReLU CUDA kernel
fused_gemm_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void bias_relu_kernel(float* output, const float* bias, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int idx = row * N + col;
        output[idx] = max(0.0f, output[idx] + bias[col]);
    }
}

torch::Tensor fused_gemm_bias_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    auto output = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(input.device()));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Perform GEMM: output = input * weight^T (CUBLAS_OP_T for weight transpose)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                N, M, K, 
                &alpha, 
                weight.data_ptr<float>(), N, 
                input.data_ptr<float>(), K, 
                &beta, 
                output.data_ptr<float>(), N);
    
    // Add bias and apply ReLU in a single kernel
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    bias_relu_kernel<<<grid, block>>>(output.data_ptr<float>(), bias.data_ptr<float>(), M, N);
    
    cublasDestroy(handle);
    return output;
}
"""

fused_gemm_bias_relu_cpp_source = "torch::Tensor fused_gemm_bias_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
fused_gemm_bias_relu = load_inline(
    name="fused_gemm_bias_relu",
    cpp_sources=fused_gemm_bias_relu_cpp_source,
    cuda_sources=fused_gemm_bias_relu_source,
    functions=["fused_gemm_bias_relu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_gemm_bias_relu

    def forward(self, x):
        return self.fused_op.fused_gemm_bias_relu_cuda(x, self.weight, self.bias)