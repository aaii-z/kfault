import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM + scaling + hardtanh + GELU CUDA kernel
fused_gemm_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

// GELU approximation constants
constexpr float GELU_COEF_A = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ float hardtanh_activation(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__global__ void fused_activation_kernel(
    float* output, 
    const float* input, 
    const float* weight, 
    const float* bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max,
    int batch_size,
    int in_features,
    int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        if (bias) {
            sum += bias[col];
        }
        
        // Apply scaling
        sum *= scaling_factor;
        
        // Apply hardtanh
        sum = hardtanh_activation(sum, hardtanh_min, hardtanh_max);
        
        // Apply GELU
        output[row * out_features + col] = gelu_activation(sum);
    }
}

torch::Tensor fused_gemm_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );
    
    // Launch the fused kernel
    fused_activation_kernel<<<grid_size, block_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        scaling_factor,
        hardtanh_min,
        hardtanh_max,
        batch_size,
        in_features,
        out_features
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}
"""

fused_gemm_activation_cpp_source = """
torch::Tensor fused_gemm_activation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max);
"""

# Compile the inline CUDA code
fused_gemm_activation = load_inline(
    name="fused_gemm_activation",
    cpp_sources=fused_gemm_activation_cpp_source,
    cuda_sources=fused_gemm_activation_source,
    functions=["fused_gemm_activation_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_gemm_activation

    def forward(self, x):
        return self.fused_op.fused_gemm_activation_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.scaling_factor, 
            self.hardtanh_min, 
            self.hardtanh_max
        )