import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + BatchNorm + GELU + ReLU
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: GEMM + BatchNorm + GELU + ReLU
// BatchNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ReLU: max(0, x)

__global__ void fused_gemm_bn_gelu_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float eps) {
    
    // Calculate output element index
    // Each thread computes one element of the output matrix (batch_size x out_features)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int row = idx / out_features;  // batch index
        int col = idx % out_features;  // feature index
        
        // 1. GEMM: Compute dot product of input row and weight column
        float sum = 0.0f;
        const float* input_row = input + row * in_features;
        const float* weight_col = weight + col * in_features;
        
        // Unroll loop for better performance (assuming in_features is large)
        for (int k = 0; k < in_features; ++k) {
            sum += input_row[k] * weight_col[k];
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // 2. BatchNorm: Normalize using running mean/var and apply affine transform
        // bn_x = (x - mean) / sqrt(var + eps)
        // y = bn_x * bn_weight + bn_bias
        float mean = bn_mean[col];
        float var = bn_var[col];
        float gamma = bn_weight[col];
        float beta = bn_bias[col];
        
        float normalized = (sum - mean) * rsqrtf(var + eps);
        float bn_out = normalized * gamma + beta;
        
        // 3. GELU activation
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float c = 0.7978845608028654f; // sqrt(2/pi)
        float x = bn_out;
        float x3 = x * x * x;
        float inner = c * (x + 0.044715f * x3);
        float gelu_out = 0.5f * x * (1.0f + tanhf(inner));
        
        // 4. ReLU activation
        float relu_out = fmaxf(0.0f, gelu_out);
        
        output[idx] = relu_out;
    }
}

torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    int batch_size,
    int in_features,
    int out_features) {
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;
    
    float eps = 1e-5f; // Default epsilon for BatchNorm
    
    // Handle optional bias
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    fused_gemm_bn_gelu_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps);
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    int batch_size,
    int in_features,
    int out_features);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_gemm_bn_gelu_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a fused GEMM + BatchNorm + GELU + ReLU.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define parameters manually to match nn.Linear and nn.BatchNorm1d
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.empty(out_features))
        self.bn_bias = nn.Parameter(torch.empty(out_features))
        
        # Register buffers for BatchNorm running stats
        self.register_buffer('running_mean', torch.empty(out_features))
        self.register_buffer('running_var', torch.empty(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.ones_(self.bn_weight)
        nn.init.zeros_(self.bn_bias)
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)
        
        self.fused_ops = fused_ops

    def forward(self, x):
        batch_size = x.size(0)
        return self.fused_ops.fused_gemm_bn_gelu_relu_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.running_mean, 
            self.running_var, 
            self.bn_weight, 
            self.bn_bias,
            batch_size,
            self.in_features,
            self.out_features
        )


import math