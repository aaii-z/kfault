import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + group norm + leaky relu + elementwise add
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_THREADS 1024

__global__ void fused_linear_gn_leakyrelu_add_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_groups,
    float eps,
    float negative_slope
) {
    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;

    if (batch_idx >= batch_size || feature_idx >= hidden_size) return;

    int group_size = hidden_size / num_groups;
    int group_id = feature_idx / group_size;
    int local_idx = feature_idx % group_size;

    // Shared memory for mean and variance reduction within group
    extern __shared__ float shared_data[];
    float* mean_shared = shared_data;
    float* var_shared = shared_data + group_size;

    // GEMM: output = input @ weight + bias
    float sum = 0.0f;
    for (int k = 0; k < input_size; k++) {
        sum += input[batch_idx * input_size + k] * weight[feature_idx * input_size + k];
    }
    sum += bias[feature_idx];
    float x = sum;

    // Store value in shared memory for this group
    mean_shared[local_idx] = x;
    __syncthreads();

    // Compute mean of the group
    float group_mean = 0.0f;
    for (int i = 0; i < group_size; i++) {
        group_mean += mean_shared[i];
    }
    group_mean /= group_size;
    __syncthreads();

    // Compute variance
    float diff = x - group_mean;
    var_shared[local_idx] = diff * diff;
    __syncthreads();

    float group_var = 0.0f;
    for (int i = 0; i < group_size; i++) {
        group_var += var_shared[i];
    }
    group_var /= group_size;
    __syncthreads();

    float inv_std = rsqrtf(group_var + eps);

    // Normalize, scale and shift
    float x_hat = (x - group_mean) * inv_std;
    x_hat = gamma[group_id] * x_hat + beta[group_id];

    // Leaky ReLU
    x_hat = fmaxf(x_hat, negative_slope * x_hat);

    // Element-wise sum: x_hat + x_hat
    x_hat = x_hat + x_hat;

    output[batch_idx * hidden_size + feature_idx] = x_hat;
}
"""

fused_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps,
    float negative_slope
);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

        # Define parameters
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        self.gamma = nn.Parameter(torch.empty(num_groups))
        self.beta = nn.Parameter(torch.empty(num_groups))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=negative_slope)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

        # Register fused operation
        self.fused_op = fused_ops

    def forward(self, x):
        return self.fused_op.fused_forward_cuda(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.num_groups, self.eps, self.negative_slope
        )