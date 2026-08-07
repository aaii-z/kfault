import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + group norm + leaky relu + elementwise add
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

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
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (col >= hidden_size) return;

    int group_id = col / (hidden_size / num_groups);
    int group_size = hidden_size / num_groups;
    int group_start = group_id * group_size;
    int local_idx = col - group_start;

    // Shared memory for group statistics
    __shared__ float mean[512];
    __shared__ float rstd[512];

    // Perform GEMM: output = input @ weight + bias
    float sum = 0.0f;
    for (int k = 0; k < input_size; k++) {
        sum += input[row * input_size + k] * weight[col * input_size + k];
    }
    sum += bias[col];
    float x = sum;

    // Compute mean for the group (within warp)
    float thread_mean = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int idx = group_start + i;
        if (row * hidden_size + idx < batch_size * hidden_size) {
            float val = (idx == col) ? x : __shfl_sync(0xFFFFFFFF, x, i);
            thread_mean += val;
        }
    }
    mean[local_idx] = thread_mean;

    __syncthreads();

    // Reduce within group
    float group_mean = 0.0f;
    for (int i = 0; i < group_size; i++) {
        group_mean += mean[i];
    }
    group_mean /= group_size;

    // Compute variance
    float thread_var = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int idx = group_start + i;
        float val = (idx == col) ? x : __shfl_sync(0xFFFFFFFF, x, i);
        float diff = val - group_mean;
        thread_var += diff * diff;
    }
    mean[local_idx] = thread_var;
    __syncthreads();

    float group_var = 0.0f;
    for (int i = 0; i < group_size; i++) {
        group_var += mean[i];
    }
    group_var /= group_size;
    rstd[local_idx] = 1.0f / sqrtf(group_var + eps);

    __syncthreads();

    // Normalize
    float x_hat = (x - group_mean) * rstd[local_idx];

    // Scale and shift
    x_hat = gamma[group_id] * x_hat + beta[group_id];

    // Leaky ReLU
    x_hat = (x_hat >= 0.0f) ? x_hat : negative_slope * x_hat;

    // Element-wise sum: x_hat + x_hat
    x_hat = x_hat + x_hat;

    output[row * hidden_size + col] = x_hat;
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

        # Register as compiled module
        self.fused_op = fused_ops

    def forward(self, x):
        return self.fused_op.fused_forward_cuda(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.num_groups, self.eps, self.negative_slope
        )