import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused matmul + swish + bias + groupnorm
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_matmul_swish_bias_gn_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ weight_gn,
    const float* __restrict__ bias_gn,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= batch_size || col >= out_features) return;

    // Matrix multiplication: output[row][col] = sum_k input[row][k] * weight[col][k]
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }

    // Swish: sum = sigmoid(sum) * sum
    float swish_out = sigmoid(sum) * sum;

    // Add bias
    float val = swish_out + bias[col];

    int group_size = out_features / num_groups;
    int group_idx = col / group_size;
    int group_offset = group_idx * group_size;

    // Compute mean within group
    float mean = 0.0f;
    for (int i = 0; i < group_size; i++) {
        int c = group_offset + i;
        float x_val = c == col ? val : 
            sigmoid(input[row * in_features + 0] * weight[c * in_features + 0]) * input[row * in_features + 0] + bias[c];
        mean += x_val;
    }
    mean /= group_size;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < group_size; i++) {
        int c = group_offset + i;
        float x_val = c == col ? val : 
            sigmoid(input[row * in_features + 0] * weight[c * in_features + 0]) * input[row * in_features + 0] + bias[c];
        float diff = x_val - mean;
        var += diff * diff;
    }
    var /= group_size;
    float inv_std = rsqrtf(var + 1e-5f);

    // Apply GroupNorm: (val - mean) * inv_std * weight_gn[group_idx] + bias_gn[group_idx]
    float result = (val - mean) * inv_std * weight_gn[group_idx] + bias_gn[group_idx];
    output[row * out_features + col] = result;
}
"""

fused_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int64_t num_groups);
"""

# Define the fused forward function in C++ to be linked properly
fused_function_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern void fused_matmul_swish_bias_gn_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ weight_gn,
    const float* __restrict__ bias_gn,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups
);

torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    int64_t num_groups) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size_x = 32;
    const int block_size_y = 8;
    dim3 block(block_size_x, block_size_y);
    dim3 grid(
        (out_features + block_size_x - 1) / block_size_x,
        (batch_size + block_size_y - 1) / block_size_y
    );

    fused_matmul_swish_bias_gn_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        weight_gn.data_ptr<float>(),
        bias_gn.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        static_cast<int>(num_groups)
    );

    return output;
}
"""

# Compile the inline CUDA module
fused_module = load_inline(
    name="fused_module",
    cpp_sources=[fused_function_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_forward_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for matmul + swish + bias + groupnorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        self.matmul_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.matmul_weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.matmul_weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_module.fused_forward_cuda(
            x, self.matmul_weight, self.bias,
            self.group_norm.weight, self.group_norm.bias,
            self.num_groups
        )