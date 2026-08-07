import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GELU + global average pooling
fused_gelu_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// GELU approximation
__device__ __forceinline__ float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float approx_coef = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + approx_coef * x * x)));
}

__global__ void fused_gelu_pool_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* partial_sums = shared_mem;

    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = ((batch * channels + channel) * spatial_size + i);
        float val = input[idx];
        sum += gelu_approx(val);
    }

    partial_sums[tid] = sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch * channels + channel] = partial_sums[0] / spatial_size;
    }
}

torch::Tensor fused_gelu_pool_cuda(torch::Tensor input) {
    CHECK_INPUT(input);

    int batch_size = input.size(0);
    int channels = input.size(1);
    int spatial_size = input.size(2) * input.size(3);

    auto output = torch::empty({batch_size, channels}, input.options());

    dim3 grid(batch_size, channels);
    int block_size = 1024;
    size_t shared_mem_size = block_size * sizeof(float);

    fused_gelu_pool_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size
    );

    return output;
}
"""

fused_gelu_pool_cpp_source = "torch::Tensor fused_gelu_pool_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_gelu_pool = load_inline(
    name="fused_gelu_pool",
    cpp_sources=fused_gelu_pool_cpp_source,
    cuda_sources=fused_gelu_pool_source,
    functions=["fused_gelu_pool_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv(x)
        x = fused_gelu_pool.fused_gelu_pool_cuda(x)
        return x