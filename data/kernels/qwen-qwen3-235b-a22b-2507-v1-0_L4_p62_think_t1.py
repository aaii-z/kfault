import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

group_norm_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void group_norm_reduction_kernel(
    const float* input,
    float* sum,
    float* sum_sq,
    int batch_size,
    int hidden_size,
    int num_groups) {
    
    int group_size = hidden_size / num_groups;
    int group_idx = blockIdx.x;
    int N = batch_size * group_size;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int batch = i / group_size;
        int channel_in_group = i % group_size;
        int channel = group_idx * group_size + channel_in_group;
        int idx = batch * hidden_size + channel;
        float x = input[idx];
        local_sum += x;
        local_sum_sq += x * x;
    }

    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_sum_sq[BLOCK_SIZE];
    shared_sum[threadIdx.x] = local_sum;
    shared_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum[group_idx] = shared_sum[0];
        sum_sq[group_idx] = shared_sum_sq[0];
    }
}

__global__ void compute_mean_inv_std_kernel(
    const float* sum,
    const float* sum_sq,
    float* mean,
    float* inv_std,
    int num_groups,
    int N,
    float eps) {
    
    int group_idx = blockIdx.x;
    float s = sum[group_idx];
    float ss = sum_sq[group_idx];
    mean[group_idx] = s / N;
    float var = (ss - s * s / N) / N;
    inv_std[group_idx] = 1.0f / sqrt(var + eps);
}

__global__ void group_norm_leaky_relu_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    const float* mean,
    const float* inv_std,
    int batch_size,
    int hidden_size,
    int num_groups,
    float negative_slope) {
    
    int group_size = hidden_size / num_groups;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    int batch = idx / hidden_size;
    int channel = idx % hidden_size;
    int group_idx = channel / group_size;
    float m = mean[group_idx];
    float i_std = inv_std[group_idx];

    float x = input[idx];
    float normalized = (x - m) * i_std;
    float gamma_val = gamma[channel];
    float beta_val = beta[channel];
    float affine = normalized * gamma_val + beta_val;
    float activated = affine > 0 ? affine : affine * negative_slope;
    output[idx] = 2.0f * activated;
}

torch::Tensor group_norm_leaky_relu_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps,
    float negative_slope) {
    
    auto batch_size = input.size(0);
    auto hidden_size = input.size(1);
    int group_size = hidden_size / num_groups;
    int N = batch_size * group_size;

    auto sum = torch::zeros(num_groups, input.options());
    auto sum_sq = torch::zeros(num_groups, input.options());
    auto mean = torch::zeros(num_groups, input.options());
    auto inv_std = torch::zeros(num_groups, input.options());
    auto output = torch::empty_like(input);

    int num_blocks_reduction = num_groups;
    group_norm_reduction_kernel<<<num_blocks_reduction, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        sum.data_ptr<float>(),
        sum_sq.data_ptr<float>(),
        batch_size,
        hidden_size,
        num_groups
    );

    compute_mean_inv_std_kernel<<<num_groups, 1>>>(
        sum.data_ptr<float>(),
        sum_sq.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        num_groups,
        N,
        eps
    );

    int total_elements = batch_size * hidden_size;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    group_norm_leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        batch_size,
        hidden_size,
        num_groups,
        negative_slope
    );

    return output;
}
"""

group_norm_leaky_relu_cpp = "torch::Tensor group_norm_leaky_relu_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps, float negative_slope);"

group_norm_leaky_relu = load_inline(
    name="group_norm_leaky_relu",
    cpp_sources=group_norm_leaky_relu_cpp,
    cuda_sources=group_norm_leaky_relu_source,
    functions=["group_norm_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        self.gn_weight = nn.Parameter(torch.ones(hidden_size))
        self.gn_bias = nn.Parameter(torch.zeros(hidden_size))
        self.custom_gn_relu = group_norm_leaky_relu

    def forward(self, x):
        x = self.fc(x)
        x = self.custom_gn_relu.group_norm_leaky_relu_cuda(x, self.gn_weight, self.gn_bias, self.num_groups, self.eps, self.negative_slope)
        return x