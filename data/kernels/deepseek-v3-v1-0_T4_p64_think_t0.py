import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activations
fused_activations_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__global__ void fused_activations_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        
        // First LeakyReLU
        if (val < 0.0f) {
            val = 0.01f * val;
        }
        // Second LeakyReLU
        if (val < 0.0f) {
            val = 0.01f * val;
        }
        // First GELU
        val = gelu(val);
        // Second GELU
        val = gelu(val);
        
        x[idx] = val;
    }
}

torch::Tensor fused_activations_cuda(torch::Tensor x) {
    auto size = x.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    fused_activations_kernel<<<blocks, threads>>>(x.data_ptr<float>(), size);
    
    return x;
}
"""

fused_activations_cpp_source = "torch::Tensor fused_activations_cuda(torch::Tensor x);"

# Compile the inline CUDA code
fused_activations = load_inline(
    name="fused_activations",
    cpp_sources=fused_activations_cpp_source,
    cuda_sources=fused_activations_source,
    functions=["fused_activations_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_activations = fused_activations

    def forward(self, x):
        x = self.linear(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = self.fused_activations.fused_activations_cuda(x)
        return x