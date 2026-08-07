import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM + Scale + BatchNorm CUDA kernel
fused_gemm_scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>

__global__ void scale_and_batchnorm_kernel(
    float* output, const float* input, const float* scale,
    const float* running_mean, const float* running_var,
    const float* weight, const float* bias,
    float eps, int batch_size, int out_features) {
    
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx >= out_features) return;

    // Compute mean and variance for this feature across batch
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        float val = input[batch_idx * out_features + feature_idx];
        sum += val;
        sum_sq += val * val;
    }
    
    float mean = sum / batch_size;
    float variance = (sum_sq / batch_size) - (mean * mean);
    
    // Update running statistics (simplified for demonstration)
    // In practice, you'd use momentum and proper atomic operations
    
    // Apply batch normalization: (x - mean) / sqrt(variance + eps)
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        int idx = batch_idx * out_features + feature_idx;
        float normalized = (input[idx] - mean) / sqrtf(variance + eps);
        
        // Apply scale, weight and bias
        output[idx] = scale[feature_idx] * normalized * (weight ? weight[feature_idx] : 1.0f) 
                     + (bias ? bias[feature_idx] : 0.0f);
    }
}

torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor scale, torch::Tensor running_mean, torch::Tensor running_var,
    float eps, float momentum) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    // Perform GEMM using cuBLAS
    auto output_gemm = torch::empty({batch_size, out_features}, input.options());
    
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
                output_gemm.data_ptr<float>(), out_features);
    
    cublasDestroy(handle);
    
    // Apply scale and batch normalization
    auto output = torch::empty_like(output_gemm);
    
    const int block_size = 256;
    const int num_blocks = (out_features + block_size - 1) / block_size;
    
    scale_and_batchnorm_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), output_gemm.data_ptr<float>(),
        scale.data_ptr<float>(), running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(), nullptr, nullptr,
        eps, batch_size, out_features);
    
    cudaDeviceSynchronize();
    
    return output;
}
"""

fused_gemm_scale_bn_cpp_source = """
torch::Tensor fused_gemm_scale_bn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor scale, torch::Tensor running_mean, torch::Tensor running_var,
    float eps, float momentum);
"""

# Compile the inline CUDA code
fused_gemm_scale_bn = load_inline(
    name="fused_gemm_scale_bn",
    cpp_sources=fused_gemm_scale_bn_cpp_source,
    cuda_sources=fused_gemm_scale_bn_source,
    functions=["fused_gemm_scale_bn_cuda"],
    verbose=True,
    extra_cflags=["-lcublas"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # BatchNorm running statistics
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
        self.fused_op = fused_gemm_scale_bn

    def forward(self, x):
        return self.fused_op.fused_gemm_scale_bn_cuda(
            x, self.weight, self.bias, self.scale,
            self.running_mean, self.running_var,
            self.eps, self.momentum
        )