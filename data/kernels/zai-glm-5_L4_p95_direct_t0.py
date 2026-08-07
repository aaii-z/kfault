import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operator
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function for GELU (approximation)
__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Fused kernel: MatMul result + Add -> Swish -> Tanh -> GELU -> Hardtanh
// Note: This kernel fuses the element-wise operations. 
// The matrix multiplication is handled by cuBLAS via torch::matmul, 
// and this kernel processes the result element-wise.
__global__ void fused_ops_kernel(const float* __restrict__ x, const float* __restrict__ add_value, float* __restrict__ out, int size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float val = x[idx];
        
        // 1. Add value (broadcasting assumed: add_value is 1D, x is 2D)
        // We use modulo indexing for broadcasting
        val += add_value[idx % out_features];
        
        // 2. Swish: x * sigmoid(x)
        float sig = 1.0f / (1.0f + expf(-val));
        val = val * sig;
        
        // 3. Tanh
        val = tanhf(val);
        
        // 4. GELU (approximation)
        val = gelu_approx(val);
        
        // 5. Hardtanh (min_val=-1, max_val=1)
        val = fmaxf(-1.0f, fminf(1.0f, val));
        
        out[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor add_value) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    int out_features = x.size(1);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        add_value.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size, 
        out_features
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor add_value);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for fused operations.
    The matrix multiplication uses PyTorch's optimized backend (cuBLAS),
    followed by a custom kernel that fuses addition, Swish, Tanh, GELU, and Hardtanh.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        # We keep the Linear layer to utilize cuBLAS for matmul
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        # Perform matrix multiplication
        x = self.matmul(x)
        # Apply fused operations via custom CUDA kernel
        x = self.fused_ops.fused_ops_cuda(x, self.add_value)
        return x