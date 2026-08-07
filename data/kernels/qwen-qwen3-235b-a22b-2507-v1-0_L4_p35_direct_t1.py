import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused operations: subtract + hardswish and mish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// HardSwish activation: x * relu6(x + 3) / 6
__device__ float hardswish(float x) {
    float offset = x + 3.0f;
    float relu6 = fminf(fmaxf(offset, 0.0f), 6.0f);
    return x * relu6 / 6.0f;
}

// Mish activation: x * tanh(softplus(x))
__device__ float mish(float x) {
    float sp = log1pf(expf(x));
    return x * tanhf(sp);
}

// Fused kernel: subtract value and apply HardSwish
__global__ void fused_subtract_hardswish_kernel(const float* input, float* output, float subtract_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - subtract_value;
        output[idx] = hardswish(x);
    }
}

// Fused kernel: apply Mish activation
__global__ void fused_mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = mish(input[idx]);
    }
}

// CUDA kernel launch wrappers
torch::Tensor fused_subtract_hardswish_cuda(torch::Tensor input, float subtract_value) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    fused_subtract_hardswish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), subtract_value, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    return output;
}

torch::Tensor fused_mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    fused_mish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_subtract_hardswish_cuda(torch::Tensor input, float subtract_value);
torch::Tensor fused_mish_cuda(torch::Tensor input);
"""

# Compile the inline CUDA kernels
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_subtract_hardswish_cuda", "fused_mish_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernels for fused operations:
    - Fused subtract + HardSwish after convolution
    - Fused Mish activation after pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        # Bind CUDA functions
        self.fused_subtract_hardswish = fused_ops.fused_subtract_hardswish_cuda
        self.fused_mish = fused_ops.fused_mish_cuda

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_subtract_hardswish(x, self.subtract_value)
        x = self.pool(x)
        x = self.fused_mish(x)
        return x