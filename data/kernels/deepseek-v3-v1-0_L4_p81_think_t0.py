import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused activation kernel
fused_activation_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_activation_kernel(
    const float* input,
    float* output,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float x = input[idx];
        // Swish: x * sigmoid(x)
        float swish = x * sigmoid(x);
        // Divide by 2
        float divided = swish * 0.5f;
        // First clamp
        float clamped1 = fminf(fmaxf(divided, -1.0f), 1.0f);
        // Tanh
        float tanh_val = tanhf(clamped1);
        // Final clamp
        float result = fminf(fmaxf(tanh_val, -1.0f), 1.0f);
        output[idx] = result;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_activation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements
    );
    
    return output;
}
"""

fused_activation_cpp = "torch::Tensor fused_activation_cuda(torch::Tensor input);"

# Load the custom CUDA extension
fused_activation_ext = load_inline(
    name='fused_activation',
    cpp_sources=fused_activation_cpp,
    cuda_sources=fused_activation_source,
    functions=['fused_activation_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.activation = fused_activation_ext.fused_activation_cuda

    def forward(self, x):
        x = self.gemm(x)
        x = self.activation(x)
        return x