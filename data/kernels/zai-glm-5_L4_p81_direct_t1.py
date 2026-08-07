import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Swish + Divide + Clamp + Tanh + Clamp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel for GEMM + Post-processing
// Each thread computes one output element.
// We use vectorized loads (float4) to maximize memory bandwidth for the GEMM part.
__global__ void fused_gemm_swish_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx < total_elements) {
        int row = idx / out_features;
        int col = idx % out_features;
        
        float sum = 0.0f;
        const float* input_row = input + row * in_features;
        const float* weight_row = weight + col * in_features; // Weight is (out_features, in_features)
        
        // Vectorized processing for GEMM
        int i = 0;
        // Process in chunks of 4
        for (; i + 3 < in_features; i += 4) {
            // Use reinterpret_cast to load float4
            float4 in_val = *reinterpret_cast<const float4*>(input_row + i);
            float4 w_val = *reinterpret_cast<const float4*>(weight_row + i);
            
            sum += in_val.x * w_val.x;
            sum += in_val.y * w_val.y;
            sum += in_val.z * w_val.z;
            sum += in_val.w * w_val.w;
        }

        // Process remaining elements
        for (; i < in_features; ++i) {
            sum += input_row[i] * weight_row[i];
        }
        
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Fused post-processing
        // Swish: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + expf(-sum));
        sum = sum * sigmoid_val;
        
        // Divide by 2.0
        sum = sum * 0.5f;
        
        // Clamp between -1.0 and 1.0
        sum = fmaxf(-1.0f, fminf(1.0f, sum));
        
        // Tanh
        sum = tanhf(sum);
        
        // Clamp between -1.0 and 1.0 (Redundant mathematically for tanh, but kept for architectural fidelity)
        sum = fmaxf(-1.0f, fminf(1.0f, sum));
        
        output[idx] = sum;
    }
}

torch::Tensor fused_gemm_swish_tanh_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    const int block_size = 256;
    int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_gemm_swish_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_gemm_swish_tanh_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_gemm_swish_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel fusing GEMM, Swish, Divide, Clamp, Tanh, and Clamp.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight and bias parameters similar to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_gemm_swish_tanh_cuda(x, self.weight, self.bias)