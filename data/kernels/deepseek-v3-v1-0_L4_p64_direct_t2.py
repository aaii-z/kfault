import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Gemm + LogSumExp + LeakyReLU + GELU
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>

constexpr float negative_slope = 0.01f;
constexpr float sqrt_2_over_pi = 0.7978845608028654f;
constexpr float gelu_coeff = 0.044715f;

__device__ float leaky_relu(float x) {
    return x > 0.0f ? x : negative_slope * x;
}

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + gelu_coeff * x * x)));
}

__global__ void fused_gemm_activation_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[k * out_features + col];
        }
        if (bias) {
            sum += bias[col];
        }
        output[row * out_features + col] = sum;
    }
}

__global__ void logsumexp_activations_kernel(
    const float* input, float* output, int batch_size, int out_features) {
    
    int row = blockIdx.x;
    
    if (row < batch_size) {
        // Find max value in row for numerical stability
        float max_val = -FLT_MAX;
        for (int col = 0; col < out_features; col++) {
            float val = input[row * out_features + col];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        // Compute sum of exponentials
        float exp_sum = 0.0f;
        for (int col = 0; col < out_features; col++) {
            exp_sum += expf(input[row * out_features + col] - max_val);
        }
        
        // Compute logsumexp
        float result = max_val + logf(exp_sum);
        
        // Apply activations
        result = leaky_relu(result);
        result = leaky_relu(result);
        result = gelu(result);
        result = gelu(result);
        
        output[row] = result;
    }
}

torch::Tensor fused_operator_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(1);
    
    auto gemm_output = torch::empty({batch_size, out_features}, input.options());
    auto final_output = torch::empty({batch_size}, input.options());
    
    // Launch GEMM kernel with proper grid and block dimensions
    dim3 gemm_block(16, 16);
    dim3 gemm_grid((out_features + gemm_block.x - 1) / gemm_block.x, 
                   (batch_size + gemm_block.y - 1) / gemm_block.y);
    
    fused_gemm_activation_kernel<<<gemm_grid, gemm_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        gemm_output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    // Launch LogSumExp + activations kernel
    logsumexp_activations_kernel<<<batch_size, 1>>>(
        gemm_output.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return final_output.unsqueeze(1);
}
"""

fused_operator_cpp_source = "torch::Tensor fused_operator_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.fused_op = fused_operator

    def forward(self, x):
        return self.fused_op.fused_operator_cuda(x, self.weight, self.bias if self.bias is not None else torch.Tensor())