import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bias_hardtanh_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bias_hardtanh_mish_kernel(
    const float* input,
    const float* bias,
    float* output,
    int size,
    int out_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int j = idx % out_features;
        float val = input[idx] + bias[j];
        // Hardtanh: clamp to [-1, 1]
        val = (val < -1.0f) ? -1.0f : (val > 1.0f) ? 1.0f : val;
        // Mish: x * tanh(softplus(x))
        float sp = log1pf(expf(val));
        float tanh_sp = tanhf(sp);
        output[idx] = val * tanh_sp;
    }
}

torch::Tensor fused_bias_hardtanh_mish_cuda(torch::Tensor input, torch::Tensor bias) {
    auto size = input.numel();
    auto out_features = input.size(-1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_bias_hardtanh_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        out_features
    );

    return output;
}
"""

fused_bias_hardtanh_mish_cpp = "torch::Tensor fused_bias_hardtanh_mish_cuda(torch::Tensor input, torch::Tensor bias);"

fused_bias_hardtanh_mish = load_inline(
    name="fused_bias_hardtanh_mish",
    cpp_sources=fused_bias_hardtanh_mish_cpp,
    cuda_sources=fused_bias_hardtanh_mish_source,
    functions=["fused_bias_hardtanh_mish_cuda"],
    verbose=False
)

groupnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void groupnorm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int num_channels,
    int num_groups,
    float eps) {
    
    int group_id = blockIdx.x;
    int sample_id = group_id / num_groups;
    int group_idx = group_id % num_groups;

    int group_size = num_channels / num_groups;
    int start_idx = sample_id * num_channels + group_idx * group_size;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < group_size; i++) {
        float x = input[start_idx + i];
        sum += x;
        sum_sq += x * x;
    }

    float mean = sum / group_size;
    float variance = (sum_sq - (sum * sum) / group_size) / group_size;
    float inv_std = 1.0f / sqrtf(variance + eps);

    for (int i = 0; i < group_size; i++) {
        float x = input[start_idx + i];
        float normalized = (x - mean) * inv_std;
        output[start_idx + i] = normalized * weight[group_idx * group_size + i] + bias[group_idx * group_size + i];
    }
}

torch::Tensor groupnorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps) {
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    auto output = torch::empty_like(input);

    int num_blocks = batch_size * num_groups;

    groupnorm_kernel<<<num_blocks, 1>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        eps
    );

    return output;
}
"""

groupnorm_cpp = "torch::Tensor groupnorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps);"

groupnorm_op = load_inline(
    name="groupnorm",
    cpp_sources=groupnorm_cpp,
    cuda_sources=groupnorm_source,
    functions=["groupnorm_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.fused_op = fused_bias_hardtanh_mish
        self.groupnorm_custom = groupnorm_op

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_bias_hardtanh_mish_cuda(x, self.bias)
        x = self.groupnorm_custom.groupnorm_cuda(x, self.groupnorm.weight, self.groupnorm.bias, self.groupnorm.num_groups, 1e-5)
        return x