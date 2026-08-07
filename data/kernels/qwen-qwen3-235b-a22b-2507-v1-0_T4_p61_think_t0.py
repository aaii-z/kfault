import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_relu_groupnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_relu_groupnorm_forward_reductions(
    const float* input,
    float* sum,
    float* sum_sq,
    int batch_size,
    int channels,
    int groups,
    int D,
    int H,
    int W
) {
    int group_idx = blockIdx.x % groups;
    int sample_idx = blockIdx.x / groups;
    if (sample_idx >= batch_size) return;

    int channels_per_group = channels / groups;
    int group_size = channels_per_group * D * H * W;

    int total_elements = group_size;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* group_data = input + sample_idx * channels * D * H * W + group_idx * channels_per_group * D * H * W;

    for (int i = tid; i < total_elements; i += stride) {
        float x = group_data[i];
        float relu_x = (x > 0.0f) ? x : 0.0f;
        local_sum += relu_x;
        local_sum_sq += relu_x * relu_x;
    }

    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sum_sq[tid] += shared_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[sample_idx * groups + group_idx] = shared_sum[0];
        sum_sq[sample_idx * groups + group_idx] = shared_sum_sq[0];
    }
}

__global__ void fused_relu_groupnorm_forward_normalization(
    const float* input,
    float* output,
    const float* mean,
    const float* var,
    const float* gamma,
    const float* beta,
    int batch_size,
    int channels,
    int groups,
    int D,
    int H,
    int W,
    float eps
) {
    int group_idx = blockIdx.x % groups;
    int sample_idx = blockIdx.x / groups;
    if (sample_idx >= batch_size) return;

    int channels_per_group = channels / groups;
    int group_size = channels_per_group * D * H * W;

    int total_elements = group_size;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* group_data = input + sample_idx * channels * D * H * W + group_idx * channels_per_group * D * H * W;
    float* group_output = output + sample_idx * channels * D * H * W + group_idx * channels_per_group * D * H * W;

    float mean_val = mean[sample_idx * groups + group_idx];
    float var_val = var[sample_idx * groups + group_idx];
    float inv_std = 1.0f / sqrtf(var_val + eps);

    for (int i = tid; i < total_elements; i += stride) {
        int c = i / (D * H * W);
        float x = group_data[i];
        float relu_x = (x > 0.0f) ? x : 0.0f;
        float normalized = (relu_x - mean_val) * inv_std;
        float gamma_val = gamma[group_idx * channels_per_group + c];
        float beta_val = beta[group_idx * channels_per_group + c];
        group_output[i] = normalized * gamma_val + beta_val;
    }
}

torch::Tensor fused_relu_groupnorm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto sum = torch::zeros({batch_size, groups}, input.options());
    auto sum_sq = torch::zeros({batch_size, groups}, input.options());

    dim3 block_size(256);
    int num_blocks = batch_size * groups;
    if (num_blocks > 0) {
        fused_relu_groupnorm_forward_reductions<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            sum.data_ptr<float>(),
            sum_sq.data_ptr<float>(),
            batch_size,
            channels,
            groups,
            D,
            H,
            W
        );
    }

    int N = D * H * W * (channels / groups);
    auto mean = sum / N;
    auto var = (sum_sq / N) - (mean * mean);

    auto output = torch::empty_like(input);

    if (num_blocks > 0) {
        fused_relu_groupnorm_forward_normalization<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            mean.data_ptr<float>(),
            var.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            batch_size,
            channels,
            groups,
            D,
            H,
            W,
            eps
        );
    }

    return output;
}
"""

fused_relu_groupnorm_cpp_source = (
    "torch::Tensor fused_relu_groupnorm_cuda("
    "torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int groups, float eps);"
)

fused_relu_groupnorm = load_inline(
    name="fused_relu_groupnorm",
    cpp_sources=fused_relu_groupnorm_cpp_source,
    cuda_sources=fused_relu_groupnorm_source,
    functions=["fused_relu_groupnorm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.fused_relu_groupnorm = fused_relu_groupnorm

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_relu_groupnorm.fused_relu_groupnorm_cuda(
            x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups, self.group_norm.eps
        )
        return x