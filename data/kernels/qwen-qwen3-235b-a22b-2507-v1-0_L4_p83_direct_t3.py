import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused min, clamp, and dropout operations
elementwise_fuse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void elementwise_fuse_kernel(
    const float* input,
    float* output,
    float min_val,
    float max_val,
    float dropout_p,
    uint64_t seed,
    int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Initialize CURAND state
    curandState state;
    curand_init(seed, idx, 0, &state);

    float x = input[idx];
    // Apply min(x, min_val)
    x = fminf(x, min_val);
    // Clamp between min_val and max_val
    x = fmaxf(x, min_val);
    x = fminf(x, max_val);

    // Apply dropout
    float rand_val = curand_uniform(&state);
    float result = (rand_val > dropout_p) ? x / (1.0f - dropout_p) : 0.0f;

    output[idx] = result;
}

torch::Tensor elementwise_fuse_cuda(torch::Tensor input, float min_val, float max_val, float dropout_p, uint64_t seed) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    elementwise_fuse_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        min_val,
        max_val,
        dropout_p,
        seed,
        size
    );

    return output;
}
"""

elementwise_fuse_cpp = """
torch::Tensor elementwise_fuse_cuda(torch::Tensor input, float min_val, float max_val, float dropout_p, uint64_t seed);
"""

# Compile the inline CUDA kernel
elementwise_fuse = load_inline(
    name="elementwise_fuse",
    cpp_sources=[elementwise_fuse_cpp],
    cuda_sources=[elementwise_fuse_source],
    functions=["elementwise_fuse_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel fusing min, clamp, and dropout.
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
        x = elementwise_fuse.elementwise_fuse_cuda(x, self.min_value, self.max_value, self.dropout_p, self.seed)
        return x