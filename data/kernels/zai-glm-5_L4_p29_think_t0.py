import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias addition and double Mish
fused_bias_mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish_func(float x) {
    // Mish activation: x * tanh(softplus(x))
    // softplus(x) = ln(1 + exp(x))
    // For numerical stability, handle large x:
    // If x > 20, exp(x) is very large, ln(1+exp(x)) approx x.
    // tanh(x) approx 1 for large x.
    // So mish(x) approx x for large x.
    float sp = (x > 20.0f) ? x : log1pf(expf(x));
    return x * tanhf(sp);
}

__global__ void fused_bias_mish_mish_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, 
    int features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * features;
    
    if (idx < total_elements) {
        int feature_idx = idx % features;
        
        // Load input and bias, add them
        float val = input[idx] + bias[feature_idx];
        
        // Apply Mish twice
        val = mish_func(val);
        val = mish_func(val);
        
        // Store result
        output[idx] = val;
    }
}

torch::Tensor fused_bias_mish_mish_cuda(torch::Tensor input, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto features = input.size(1);
    auto output = torch::empty_like(input);
    
    int total_elements = batch_size * features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bias_mish_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        features
    );
    
    return output;
}
"""

fused_bias_mish_mish_cpp_source = (
    "torch::Tensor fused_bias_mish_mish_cuda(torch::Tensor input, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_mish_op = load_inline(
    name="fused_mish_op",
    cpp_sources=fused_bias_mish_mish_cpp_source,
    cuda_sources=fused_bias_mish_mish_source,
    functions=["fused_bias_mish_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies Mish, and applies Mish again.
    Fused bias addition and Mish operations into a single custom CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # We separate the bias from the Linear layer to fuse it with Mish
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # Define bias as a separate parameter
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize bias similar to nn.Linear
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_mish_op = fused_mish_op

    def forward(self, x):
        # Matrix multiplication (using optimized cuBLAS)
        x = self.linear(x)
        
        # Fused Bias + Mish + Mish
        x = self.fused_mish_op.fused_bias_mish_mish_cuda(x, self.bias)
        
        return x