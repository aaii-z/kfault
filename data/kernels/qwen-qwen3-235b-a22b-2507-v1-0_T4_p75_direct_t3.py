import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused GEMM + GroupNorm + Min + Bias
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel 1: Fused GEMM + GroupNorm
__global__ void gemm_groupnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias_gemm,
    const float* __restrict__ weight_gn,
    const float* __restrict__ bias_gn,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    float eps
) {
    int feature_id = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;

    if (feature_id >= out_features) return;

    // GEMM: y[b][f] = sum_k (x[b][k] * w[f][k]) + bias_gemm[f]
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[batch_id * in_features + k] * weight[feature_id * in_features + k];
    }
    sum += bias_gemm[feature_id];

    // Group Normalization
    int group_size = out_features / num_groups;
    int group_id = feature_id / group_size;
    int local_id = feature_id % group_size;

    // Shared memory for mean and variance computation within group
    extern __shared__ float s_data[];
    float* s_mean = s_data;
    float* s_var = s_data + num_groups;
    float* s_group = s_data + 2 * num_groups;

    s_group[local_id] = sum;
    __syncthreads();

    // Only one thread per group computes mean and variance
    if (local_id == 0) {
        float group_sum = 0.0f;
        for (int i = 0; i < group_size; i++) {
            group_sum += s_group[i];
        }
        s_mean[group_id] = group_sum / group_size;

        float var = 0.0f;
        for (int i = 0; i < group_size; i++) {
            float diff = s_group[i] - s_mean[group_id];
            var += diff * diff;
        }
        s_var[group_id] = rsqrtf(var / group_size + eps);
    }
    __syncthreads();

    float mean = s_mean[group_id];
    float inv_std = s_var[group_id];
    float out_val = (sum - mean) * inv_std * weight_gn[feature_id] + bias_gn[feature_id];

    output[batch_id * out_features + feature_id] = out_val;
}

// Kernel 2: Min over dim=1 (sequence dim) and add bias
__global__ void min_and_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_features) return;

    float min_val = x[idx];
    for (int b = 1; b < batch_size; b++) {
        float val = x[b * out_features + idx];
        min_val = fminf(min_val, val);
    }

    output[idx] = min_val + bias[idx];
}

// Host function
torch::Tensor fused_gemm_groupnorm_min_bias(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_gemm,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    torch::Tensor bias_out,
    int64_t num_groups
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias_gemm);
    CHECK_INPUT(weight_gn);
    CHECK_INPUT(bias_gn);
    CHECK_INPUT(bias_out);

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    const float eps = 1e-5f;

    auto options = input.options();
    auto gemm_output = torch::empty({batch_size, out_features}, options);

    const int block_size = 256;
    dim3 grid_gn((out_features + block_size - 1) / block_size, batch_size);
    size_t shared_mem = (2 * num_groups + out_features / num_groups) * sizeof(float);

    gemm_groupnorm_kernel<<<grid_gn, block_size, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_gemm.data_ptr<float>(),
        weight_gn.data_ptr<float>(),
        bias_gn.data_ptr<float>(),
        gemm_output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        eps
    );

    // Second kernel: min over dim=1 (batch already reduced) and add bias
    auto min_output = torch::empty({out_features}, options);
    int num_blocks = (out_features + block_size - 1) / block_size;
    min_and_bias_kernel<<<num_blocks, block_size>>>(
        gemm_output.data_ptr<float>(),
        bias_out.data_ptr<float>(),
        min_output.data_ptr<float>(),
        batch_size,
        out_features
    );

    // Reshape to (1, out_features, 1, 1)
    return min_output.view({1, out_features, 1, 1});
}
"""

fused_cpp_source = """
torch::Tensor fused_gemm_groupnorm_min_bias(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_gemm,
    torch::Tensor weight_gn,
    torch::Tensor bias_gn,
    torch::Tensor bias_out,
    int64_t num_groups);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_groupnorm_min_bias",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_groupnorm_min_bias"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    """
    Optimized model using fused CUDA kernel for GEMM, GroupNorm, Min, and Bias.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_gemm = nn.Parameter(torch.randn(out_features))
        self.weight_gn = nn.Parameter(torch.ones(out_features))
        self.bias_gn = nn.Parameter(torch.zeros(out_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_gemm_groupnorm_min_bias(
            x, self.weight, self.bias_gemm, self.weight_gn, self.bias_gn, self.bias.squeeze(), self.num_groups
        )