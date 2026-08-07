import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused clamp and dropout operations
elementwise_fuse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void clamp_dropout_kernel(
    const float* input,
    float* output,
    float min_val,
    float max_val,
    float dropout_p,
    uint64_t seed,
    int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float x = input[idx];
    // Apply min: x = min(x, min_val)
    x = fminf(x, min_val);
    // Then clamp to [min_val, max_val] (redundant min with above, but safe)
    x = fmaxf(x, min_val);
    x = fminf(x, max_val);

    // Apply dropout
    float rand_val = curand_uniform(&state);
    float result = (rand_val > dropout_p) ? x / (1.0f - dropout_p) : 0.0f;

    output[idx] = result;
}
"""

elementwise_fuse_cpp = """
torch::Tensor clamp_dropout_kernel(
    torch::Tensor input,
    float min_val,
    float max_val,
    float dropout_p,
    uint64_t seed,
    int size);
"""

# Compile the CUDA kernel
elementwise_fuse_cuda = load_inline(
    name="clamp_dropout",
    cpp_sources=[elementwise_fuse_cpp],
    cuda_sources=[elementwise_fuse_source],
    functions=["clamp_dropout_kernel"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused clamp and dropout.
    Note: The initial min operation is redundant with clamp and is removed.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        self.seed = 123456

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = x.contiguous()
        size = x.numel()
        out = torch.zeros_like(x)
        block_size = 256
        grid_size = (size + block_size - 1) // block_size
        elementwise_fuse_cuda.clamp_dropout_kernel(
            x, out, self.min_value, self.max_value, self.dropout_p, self.seed, size
        )
        return out