import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
# Fusing: Swish, Divide by 2, Clamp(-1, 1), Tanh, Clamp(-1, 1)
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float apply_ops(float val) {
    // Swish: x * sigmoid(x)
    val = val * (1.0f / (1.0f + expf(-val)));
    
    // Divide by 2.0
    val = val * 0.5f;
    
    // Clamp between -1.0 and 1.0
    val = fmaxf(-1.0f, fminf(1.0f, val));
    
    // Tanh
    val = tanhf(val);
    
    // Clamp between -1.0 and 1.0
    // Note: tanh output is strictly within (-1, 1), so this clamp is mathematically a no-op.
    // We omit it for performance.
    
    return val;
}

__global__ void fused_ops_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Using vectorized load/store (float4) for better memory bandwidth utilization
    // Each thread processes 4 elements.
    int idx4 = idx * 4;
    
    if (idx4 < size) {
        // Ensure we don't go out of bounds (though size is multiple of 4 here)
        if (idx4 + 3 < size) {
            float4 in = reinterpret_cast<const float4*>(input)[idx];
            float4 out;
            
            out.x = apply_ops(in.x);
            out.y = apply_ops(in.y);
            out.z = apply_ops(in.z);
            out.w = apply_ops(in.w);
            
            reinterpret_cast<float4*>(output)[idx] = out;
        } else {
            // Handle tail elements if size is not a multiple of 4
            for (int i = 0; i < 4; ++i) {
                if (idx4 + i < size) {
                    output[idx4 + i] = apply_ops(input[idx4 + i]);
                }
            }
        }
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto out = torch::empty_like(input);

    const int block_size = 256;
    // Adjust grid size: each thread handles 4 elements
    const int num_blocks = (size + block_size * 4 - 1) / (block_size * 4);

    fused_ops_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_ops.fused_ops_cuda(x)
        return x