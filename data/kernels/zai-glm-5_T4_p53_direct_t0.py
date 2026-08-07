import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM, scaling, hardtanh, and GELU
fused_gemm_act_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Define constants for GELU approximation
#define GELU_CONST 0.7978845608028654f
#define GELU_COEF 0.044715f

// Helper function for GELU activation
__device__ __forceinline__ float gelu(float x) {
    // Using the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float cdf = 0.5f * x * (1.0f + tanhf(GELU_CONST * (x + GELU_COEF * x * x * x)));
    return cdf;
}

// Helper function for Hardtanh
__device__ __forceinline__ float hardtanh(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, x));
}

// Kernel for fused operations
__global__ void fused_gemm_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int N, int K,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max) {
    
    // Calculate output element coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product for GEMM
        for (int k = 0; k < K; ++k) {
            sum += input[row * K + k] * weight[col * K + k];
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Apply scaling
        sum *= scaling_factor;
        
        // Apply Hardtanh
        sum = hardtanh(sum, hardtanh_min, hardtanh_max);
        
        // Apply GELU
        sum = gelu(sum);
        
        // Store result
        output[row * N + col] = sum;
    }
}

torch::Tensor fused_gemm_act_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max) {
    
    // Get dimensions
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    // Allocate output tensor
    auto output = torch::zeros({M, N}, input.options());
    
    // Set up kernel launch parameters
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    
    // Launch kernel
    fused_gemm_act_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        M, N, K,
        scaling_factor,
        hardtanh_min,
        hardtanh_max);
    
    return output;
}
"""

fused_gemm_act_cpp_source = (
    "torch::Tensor fused_gemm_act_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor, float hardtanh_min, float hardtanh_max);"
)

# Compile the inline CUDA code
fused_gemm_act = load_inline(
    name="fused_gemm_act",
    cpp_sources=fused_gemm_act_cpp_source,
    cuda_sources=fused_gemm_act_source,
    functions=["fused_gemm_act_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for fused GEMM, scaling, hardtanh, and GELU.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_gemm_act = fused_gemm_act
        
        # Initialize weights and bias similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_gemm_act.fused_gemm_act_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.scaling_factor, 
            self.hardtanh_min, 
            self.hardtanh_max
        )