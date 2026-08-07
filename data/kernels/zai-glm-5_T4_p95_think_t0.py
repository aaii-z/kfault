import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
# Fuses: Add, Swish, Tanh, GELU, Hardtanh
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_ops_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ add_value, 
    float* __restrict__ output, 
    int total_elements, 
    int add_value_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Vectorized processing using float4 for memory coalescing
    // The problem size (1024*8192) is divisible by 4, and 8192 is divisible by 4.
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    const float4* add_val4 = reinterpret_cast<const float4*>(add_value);
    
    int num_vec4 = total_elements / 4;
    int add_val_vec_size = add_value_size / 4;
    
    for (int i = idx; i < num_vec4; i += stride) {
        float4 in_val = input4[i];
        // Broadcast add_value: i % add_val_vec_size handles the row-wise broadcasting
        float4 add_v = add_val4[i % add_val_vec_size];
        float4 out_val;
        
        // Process element x
        float v0 = in_val.x + add_v.x;
        // Process element y
        float v1 = in_val.y + add_v.y;
        // Process element z
        float v2 = in_val.z + add_v.z;
        // Process element w
        float v3 = in_val.w + add_v.w;
        
        // Swish: x * sigmoid(x)
        v0 = v0 * (1.0f / (1.0f + expf(-v0)));
        v1 = v1 * (1.0f / (1.0f + expf(-v1)));
        v2 = v2 * (1.0f / (1.0f + expf(-v2)));
        v3 = v3 * (1.0f / (1.0f + expf(-v3)));
        
        // Tanh
        v0 = tanhf(v0);
        v1 = tanhf(v1);
        v2 = tanhf(v2);
        v3 = tanhf(v3);
        
        // GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
        // 1/sqrt(2) approx 0.7071067811865475
        v0 = 0.5f * v0 * (1.0f + erff(v0 * 0.7071067811865475f));
        v1 = 0.5f * v1 * (1.0f + erff(v1 * 0.7071067811865475f));
        v2 = 0.5f * v2 * (1.0f + erff(v2 * 0.7071067811865475f));
        v3 = 0.5f * v3 * (1.0f + erff(v3 * 0.7071067811865475f));
        
        // Hardtanh: clamp(x, -1, 1)
        v0 = fmaxf(-1.0f, fminf(1.0f, v0));
        v1 = fmaxf(-1.0f, fminf(1.0f, v1));
        v2 = fmaxf(-1.0f, fminf(1.0f, v2));
        v3 = fmaxf(-1.0f, fminf(1.0f, v3));
        
        out_val.x = v0;
        out_val.y = v1;
        out_val.z = v2;
        out_val.w = v3;
        
        output4[i] = out_val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor add_value) {
    auto total_elements = input.numel();
    auto out = torch::empty_like(input);
    
    int add_value_size = add_value.size(0);
    
    const int block_size = 256;
    int num_vec4 = total_elements / 4;
    const int num_blocks = (num_vec4 + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        add_value.data_ptr<float>(), 
        out.data_ptr<float>(), 
        total_elements, 
        add_value_size
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor add_value);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication followed by a fused kernel
    for addition, Swish, Tanh, GELU, and Hardtanh.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_ops.fused_ops_cuda(x, self.add_value)
        return x