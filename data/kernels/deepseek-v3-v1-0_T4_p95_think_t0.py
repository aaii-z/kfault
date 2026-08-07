import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for the activation sequence
fused_activation_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__device__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_activation_kernel(
    const float* input,
    const float* add_value,
    float* output,
    int num_elements,
    int feature_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int feature_idx = idx % feature_dim;
        float val = input[idx] + add_value[feature_idx];
        
        // Swish: sigmoid(x) * x
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        val = sigmoid_val * val;
        
        // Tanh
        val = tanhf(val);
        
        // GELU
        val = gelu_approx(val);
        
        // Hardtanh
        val = fmaxf(-1.0f, fminf(1.0f, val));
        
        output[idx] = val;
    }
}

torch::Tensor fused_activation_cuda(
    torch::Tensor input,
    torch::Tensor add_value
) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    int feature_dim = input.size(-1);
    
    const int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_activation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        add_value.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        feature_dim
    );
    
    return output;
}
"""

fused_activation_cpp_source = """
torch::Tensor fused_activation_cuda(torch::Tensor input, torch::Tensor add_value);
"""

# Compile the inline CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_activation.fused_activation_cuda(x, self.add_value)
        return x