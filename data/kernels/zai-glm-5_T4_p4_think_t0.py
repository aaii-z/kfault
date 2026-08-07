import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Double Mish
double_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float mish_func(float x) {
    // Mish(x) = x * tanh(softplus(x))
    // softplus(x) = ln(1 + exp(x))
    // For numerical stability:
    // If x is large, softplus(x) approx x, tanh(x) approx 1, mish(x) approx x.
    // If x > 20, exp(x) is huge, so we use the approximation.
    float sp = x;
    if (x <= 20.0f) {
        sp = log1pf(expf(x));
    }
    return x * tanhf(sp);
}

__global__ void double_mish_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process data in chunks of 4 using float4 for vectorized memory access
    int vec_size = size / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* out_vec = reinterpret_cast<float4*>(out);

    for (int i = idx; i < vec_size; i += stride) {
        float4 val = x_vec[i];
        val.x = mish_func(mish_func(val.x));
        val.y = mish_func(mish_func(val.y));
        val.z = mish_func(mish_func(val.z));
        val.w = mish_func(mish_func(val.w));
        out_vec[i] = val;
    }

    // Handle remaining elements (tail)
    int tail_start = vec_size * 4;
    for (int i = tail_start + idx; i < size; i += stride) {
        out[i] = mish_func(mish_func(x[i]));
    }
}

torch::Tensor double_mish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    // Calculate grid size to cover the vectorized elements
    int num_elements_vec = (size + 3) / 4; 
    const int num_blocks = (num_elements_vec + block_size - 1) / block_size;

    double_mish_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size
    );

    return out;
}
"""

double_mish_cpp_source = (
    "torch::Tensor double_mish_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for fused Double Mish
double_mish_op = load_inline(
    name="double_mish",
    cpp_sources=double_mish_cpp_source,
    cuda_sources=double_mish_source,
    functions=["double_mish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution followed by a fused Double Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.double_mish = double_mish_op

    def forward(self, x):
        x = self.conv(x)
        x = self.double_mish.double_mish_cuda(x)
        return x