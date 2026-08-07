import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + GroupNorm + Swish + Multiply + Swish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_gemm_groupnorm_swish_multiply_swish(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gamma,
    const float* beta,
    const float* multiply_weight,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= out_features) return;

    // GEMM: compute output[row][col] = dot(input[row, :], weight[col, :]) + bias[col]
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];
    float x = sum;

    int group_size = out_features / num_groups;
    int group_id = col / group_size;
    int group_offset = col % group_size;

    // Shared memory to store the entire group's GEMM outputs for GroupNorm
    extern __shared__ float shared[];
    float* shared_group = shared;

    // Each thread in the group loads its value
    if (group_offset < group_size) {
        shared_group[group_offset] = x;
    }
    __syncthreads();

    // Compute mean and variance within the group
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < group_size; ++i) {
        float val = shared_group[i];
        mean += val;
        var += val * val;
    }
    mean /= group_size;
    var = var / group_size - mean * mean;
    var = fmaxf(var, 1e-5f);
    float inv_std = rsqrtf(var);

    // Normalize, apply gamma (scale) and beta (bias)
    float norm_x = (x - mean) * inv_std;
    float y = gamma[col] * norm_x + beta[col];

    // First Swish: y * sigmoid(y)
    float sigmoid_y = 1.0f / (1.0f + expf(-fmaxf(-50.0f, fminf(50.0f, y))));
    float swish1 = y * sigmoid_y;

    // Multiply by learnable weight
    float mul = swish1 * multiply_weight[col];

    // Second Swish: mul * sigmoid(mul)
    float sigmoid_mul = 1.0f / (1.0f + expf(-fmaxf(-50.0f, fminf(50.0f, mul))));
    float swish2 = mul * sigmoid_mul;

    output[row * out_features + col] = swish2;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor multiply_weight
);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + GroupNorm + Swish + Multiply + Swish in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

        self.fused_op = fused_op

    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        # Launch kernel with dynamic shared memory
        block_size = 256
        grid_x = (self.out_features + block_size - 1) // block_size
        grid = (grid_x, x.size(0))
        shared_mem = (self.out_features // self.num_groups) * sizeof(float)

        return self.fused_op.fused_op_cuda(
            x,
            self.weight,
            self.bias,
            self.gamma,
            self.beta,
            self.multiply_weight
        )