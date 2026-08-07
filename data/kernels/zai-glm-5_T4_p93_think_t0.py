import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: Add, Min, GELU, Mul
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Constant for GELU (1 / sqrt(2))
constexpr float GELU_CONST = 0.70710678118654752440f;

__device__ __forceinline__ float gelu_activation(float x) {
    // Exact GELU implementation: x * 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + erff(x * GELU_CONST));
}

__global__ void fused_ops_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                 int size, float add_value, float multiply_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process data in float4 chunks for vectorized memory access
    int vec_size = size / 4;
    int remainder = size % 4;

    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);

    for (int i = idx; i < vec_size; i += stride) {
        float4 val = input_vec[i];

        // Process x
        val.x += add_value;
        val.x = fminf(val.x, 0.0f);
        val.x = gelu_activation(val.x);
        val.x *= multiply_value;

        // Process y
        val.y += add_value;
        val.y = fminf(val.y, 0.0f);
        val.y = gelu_activation(val.y);
        val.y *= multiply_value;

        // Process z
        val.z += add_value;
        val.z = fminf(val.z, 0.0f);
        val.z = gelu_activation(val.z);
        val.z *= multiply_value;

        // Process w
        val.w += add_value;
        val.w = fminf(val.w, 0.0f);
        val.w = gelu_activation(val.w);
        val.w *= multiply_value;

        output_vec[i] = val;
    }

    // Handle remaining elements
    if (idx < remainder) {
        int rem_idx = vec_size * 4 + idx;
        float val = input[rem_idx];
        val += add_value;
        val = fminf(val, 0.0f);
        val = gelu_activation(val);
        val *= multiply_value;
        output[rem_idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, float add_value, float multiply_value) {
    auto size = input.numel();
    auto out = torch::empty_like(input);

    const int block_size = 256;
    // Calculate grid size to cover the vectorized elements
    int num_vectors = (size + 3) / 4;
    int num_blocks = (num_vectors + block_size - 1) / block_size;
    
    // Ensure we don't launch too many blocks if input is small, though logic handles it
    // For large inputs like 128x128x130x130, we will launch sufficient blocks.

    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), out.data_ptr<float>(), size, add_value, multiply_value
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, float add_value, float multiply_value);"
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
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution followed by a fused custom CUDA kernel
    for Add -> Min(0) -> GELU -> Mul operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.add_value, self.multiply_value)
        return x