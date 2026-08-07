import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d - Subtract - HardSwish - MaxPool2d - Mish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// HardSwish activation: x * relu6(x + 3) / 6
__device__ float hardswish(float x) {
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6 / 6.0f;
}

// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
__device__ float mish(float x) {
    float sp = log1pf(expf(x));
    return x * tanhf(sp);
}

// Fused kernel: Conv2d -> Subtract -> HardSwish -> MaxPool2d -> Mish
// We assume the convolution is done first, then pointwise ops and pooling.
// However, to fully fuse conv and activation, we use implicit GEMM style or direct fusion.
// Here we do a simplified version: we assume conv output is already computed (via cuDNN),
// and we fuse the rest: Subtract -> HardSwish -> MaxPool2d -> Mish.
// But to maximize speedup, we will write a custom kernel for the entire sequence
// after the convolution (which we leave to PyTorch cuDNN for now), but fuse the rest.

// However, to get maximum benefit, we write a fused kernel for:
// (1) Post-conv pointwise: subtract, hardswish
// (2) MaxPool2d with Mish applied after pooling

// But note: MaxPool2d is not easily fused with conv unless we do custom conv+pool.
// Instead, we fuse the pointwise operations: subtract + hardswish + (pooling) + mish
// However, pooling requires reduction, so we do:
// Step 1: Conv (use PyTorch's optimized conv)
// Step 2: Custom kernel: subtract + hardswish
// Step 3: MaxPool2d (use PyTorch's optimized pool)
// Step 4: Custom kernel: mish

// But better: fuse subtract + hardswish into one kernel (element-wise), and mish separately.
// Even better: if we can fuse all non-conv operations into one stream.

// Let's create a fused kernel for: subtract + hardswish (applied on conv output)
// And another fused kernel for: mish (applied post-pool)

// Kernel 1: fused_subtract_hardswish
extern "C" __global__ void fused_subtract_hardswish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float subtract_value,
    const int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx] - subtract_value;
    output[idx] = hardswish(x);
}

// Kernel 2: fused_mish
extern "C" __global__ void fused_mish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = mish(input[idx]);
}

// Host wrapper for fused_subtract_hardswish
torch::Tensor fused_subtract_hardswish_cuda(torch::Tensor input, float subtract_value) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_subtract_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        subtract_value,
        size
    );

    cudaError_t err = cudaGetLastError();
    at::cuda::detail::checkCudaError(err);

    return output;
}

// Host wrapper for fused_mish
torch::Tensor fused_mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    cudaError_t err = cudaGetLastError();
    at::cuda::detail::checkCudaError(err);

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
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
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