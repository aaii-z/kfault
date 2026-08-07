import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused subtraction and Mish activation
fused_sub_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float mish_activation(float x) {
    // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    // For numerical stability and performance:
    // If x is large, softplus(x) ~ x, tanh(x) ~ 1, so mish(x) ~ x.
    if (x > 20.0f) {
        return x;
    }
    // log1pf(expf(x)) is stable for x <= 0.
    // For 0 < x <= 20, expf(x) is within float range.
    float sp = log1pf(expf(x));
    return x * tanhf(sp);
}

__global__ void fused_sub_mish_kernel(const float* __restrict__ input, float* __restrict__ output, int size, float sub_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        float val = input[i];
        val -= sub_val;
        output[i] = mish_activation(val);
    }
}

torch::Tensor fused_sub_mish_cuda(torch::Tensor input, float subtract_value_1, float subtract_value_2) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    // Combine subtractions into a single operation
    float total_sub = subtract_value_1 + subtract_value_2;

    const int block_size = 256;
    // Determine grid size. Using a cap to prevent exceeding max grid dimensions if size is extremely large,
    // though grid-stride loop handles the logic.
    int num_blocks = (size + block_size - 1) / block_size;
    
    // Cap number of blocks to a reasonable number (e.g., 65535) if necessary, 
    // but for typical sizes, standard calculation is fine.
    // Here we use grid-stride loop so we can launch fewer blocks if we wanted, 
    // but launching enough blocks to cover size is standard.
    // Let's limit blocks to max grid dim x just in case, though 5e8 elements / 256 ~ 2e6 blocks is fine.
    // Actually, 65535 is the limit for grid dim x on older hardware, modern is much higher (2^31-1).
    // We'll stick to the calculated number but ensure it doesn't exceed a safe limit if we were strict.
    // For this problem size, calculated blocks is ~2M, which is fine.
    
    fused_sub_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, 
        total_sub
    );

    return output;
}
"""

fused_sub_mish_cpp_source = (
    "torch::Tensor fused_sub_mish_cuda(torch::Tensor input, float subtract_value_1, float subtract_value_2);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_sub_mish",
    cpp_sources=fused_sub_mish_cpp_source,
    cuda_sources=fused_sub_mish_source,
    functions=["fused_sub_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, followed by a fused subtraction and Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # Use the custom CUDA kernel for fused subtraction and Mish
        x = self.fused_ops.fused_sub_mish_cuda(x, self.subtract_value_1, self.subtract_value_2)
        return x