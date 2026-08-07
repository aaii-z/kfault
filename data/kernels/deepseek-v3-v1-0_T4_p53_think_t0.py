import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, scaling, Hardtanh, and GELU
fused_gemm_scale_hardtanh_gelu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#define BLOCK_SIZE 16

__device__ float gelu_approximation(float x) {
    return 0.5 * x * (1.0 + tanhf(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

__global__ void fused_gemm_scale_hardtanh_gelu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int M, int N, int K,
    float scaling_factor, float hardtanh_min, float hardtanh_max) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += input[row * K + k] * weight[col * K + k];
        }
        if (bias) {
            sum += bias[col];
        }
        
        sum *= scaling_factor;
        
        if (sum < hardtanh_min) sum = hardtanh_min;
        if (sum > hardtanh_max) sum = hardtanh_max;
        
        sum = gelu_approximation(sum);
        
        output[row * N + col] = sum;
    }
}

torch::Tensor fused_gemm_scale_hardtanh_gelu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, float hardtanh_min, float hardtanh_max) {
    
    auto M = input.size(0);
    auto K = input.size(1);
    auto N = weight.size(0);
    
    auto output = torch::empty({M, N}, input.options());
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    fused_gemm_scale_hardtanh_gelu_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), M, N, K,
        scaling_factor, hardtanh_min, hardtanh_max);
        
    return output;
}
"""

fused_gemm_scale_hardtanh_gelu_cpp_source = """
torch::Tensor fused_gemm_scale_hardtanh_gelu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, float hardtanh_min, float hardtanh_max);
"""

# Compile the inline CUDA code
fused_gemm_scale_hardtanh_gelu = load_inline(
    name="fused_gemm_scale_hardtanh_gelu",
    cpp_sources=fused_gemm_scale_hardtanh_gelu_cpp_source,
    cuda_sources=fused_gemm_scale_hardtanh_gelu_source,
    functions=["fused_gemm_scale_hardtanh_gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_gemm_scale_hardtanh_gelu.fused_gemm_scale_hardtanh_gelu_cuda(
            x, self.weight, self.bias,
            self.scaling_factor, self.hardtanh_min, self.hardtanh_max
        )