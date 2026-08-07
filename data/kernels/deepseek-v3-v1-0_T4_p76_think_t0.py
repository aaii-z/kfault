import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM + bias + ReLU CUDA kernel
fused_gemm_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void bias_relu_kernel(float* output, const float* bias, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int col = idx % out_features;
        float value = output[idx] + bias[col];
        output[idx] = value > 0 ? value : 0.0f;
    }
}

torch::Tensor fused_gemm_bias_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Use cuBLAS for optimized matrix multiplication
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    // Fused bias add and ReLU
    const int threads_per_block = 256;
    const int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    bias_relu_kernel<<<num_blocks, threads_per_block>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        out_features
    );
    
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
    verbose=True,
    extra_cflags=["-lcublas"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_gemm_bias_relu

    def forward(self, x):
        return self.fused_op.fused_gemm_bias_relu_cuda(x, self.weight, self.bias)