import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Mish and Tanh activation
# Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
# We fuse Mish and Tanh: out = tanh(Mish(x))
fused_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper function to compute fused Mish and Tanh
// Computes tanh(x * tanh(softplus(x)))
__device__ __forceinline__ float mish_tanh_op(float x) {
    // Compute softplus: ln(1 + exp(x))
    // Use log1p(exp(x)) for numerical stability
    float sp;
    if (x > 20.0f) {
        sp = x; // Approximation for large x to avoid overflow in exp
    } else {
        sp = log1pf(expf(x));
    }
    
    // Compute Mish: x * tanh(softplus(x))
    float mish_val = x * tanhf(sp);
    
    // Compute Tanh of the result
    return tanhf(mish_val);
}

__global__ void fused_mish_tanh_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4; // Each thread processes 4 elements (vectorized)

    if (idx + 3 < size) {
        // Load 4 floats from global memory
        float4 in_val = reinterpret_cast<const float4*>(input)[tid];
        float4 out_val;
        
        // Compute fused operation for each element
        out_val.x = mish_tanh_op(in_val.x);
        out_val.y = mish_tanh_op(in_val.y);
        out_val.z = mish_tanh_op(in_val.z);
        out_val.w = mish_tanh_op(in_val.w);
        
        // Store 4 floats to global memory
        reinterpret_cast<float4*>(output)[tid] = out_val;
    } else {
        // Handle remaining elements (tail)
        for (int i = 0; i < 4; ++i) {
            int cur_idx = idx + i;
            if (cur_idx < size) {
                output[cur_idx] = mish_tanh_op(input[cur_idx]);
            }
        }
    }
}

torch::Tensor fused_mish_tanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    // Calculate grid size based on vectorized processing (4 elements per thread)
    int num_elements = (size + 3) / 4;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_mish_tanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size
    );

    return out;
}
"""

fused_mish_tanh_cpp_source = "torch::Tensor fused_mish_tanh_cuda(torch::Tensor x);"

# Compile the inline CUDA code
fused_mish_tanh_op = load_inline(
    name="fused_mish_tanh",
    cpp_sources=fused_mish_tanh_cpp_source,
    cuda_sources=fused_mish_tanh_source,
    functions=["fused_mish_tanh_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution followed by a fused Mish and Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_op = fused_mish_tanh_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv(x)
        # Use the custom fused CUDA kernel for Mish + Tanh
        x = self.fused_op.fused_mish_tanh_cuda(x)
        return x