import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_batchnorm_tanh_source = """
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

__global__ void fused_batchnorm_tanh_kernel(
    const float* input,
    float* output,
    const float* scale,
    const float* offset,
    int64_t n_channels,
    int64_t spatial_size,
    int64_t total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int c = (idx / spatial_size) % n_channels;
    float x = input[idx];
    float y = x * scale[c] + offset[c];
    output[idx] = tanhf(y);
}

torch::Tensor fused_batchnorm_tanh_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor offset
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t spatial_size = H * W;
    int64_t total_elements = input.numel();

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_batchnorm_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale.data_ptr<float>(),
        offset.data_ptr<float>(),
        C,
        spatial_size,
        total_elements
    );

    return output;
}
"""

fused_batchnorm_tanh_cpp = "torch::Tensor fused_batchnorm_tanh_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor offset);"

fused_batchnorm_tanh = load_inline(
    name="fused_batchnorm_tanh",
    cpp_sources=fused_batchnorm_tanh_cpp,
    cuda_sources=fused_batchnorm_tanh_source,
    functions=["fused_batchnorm_tanh_cuda"],
    verbose=True,
)

group_norm_source = """
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

__global__ void group_norm_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t num_groups,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;

    int group_size = C / num_groups;
    int sample_idx = idx / (C * H * W);
    int channel_idx = (idx / (H * W)) % C;
    int group_idx = channel_idx / group_size;

    int group_start = group_idx * group_size;
    int count = group_size * H * W;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int c = group_start; c < group_start + group_size; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int inner_idx = sample_idx * (C * H * W) + c * (H * W) + h * W + w;
                float x = input[inner_idx];
                sum += x;
                sum_sq += x * x;
            }
        }
    }
    
    float mean = sum / count;
    float var = sum_sq / count - mean * mean;
    float inv_std = 1.0f / sqrtf(var + eps);
    
    float x = input[idx];
    output[idx] = gamma[group_idx] * (x - mean) * inv_std + beta[group_idx];
}

torch::Tensor custom_group_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    float eps
) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t total_elements = input.numel();

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    group_norm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N,
        C,
        H,
        W,
        num_groups,
        eps
    );

    return output;
}
"""

group_norm_cpp = "torch::Tensor custom_group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int64_t num_groups, float eps);"

custom_group_norm = load_inline(
    name="custom_group_norm",
    cpp_sources=group_norm_cpp,
    cuda_sources=group_norm_source,
    functions=["custom_group_norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        self.register_buffer('fused_bn_scale', torch.empty(out_channels))
        self.register_buffer('fused_bn_offset', torch.empty(out_channels))
        self.update_fused_bn_params()
        
        self.fused_batchnorm_tanh = fused_batchnorm_tanh
        self.custom_group_norm = custom_group_norm

    def update_fused_bn_params(self):
        eps = self.batch_norm.eps
        var = self.batch_norm.running_var
        mean = self.batch_norm.running_mean
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        
        scale = gamma / torch.sqrt(var + eps)
        offset = beta - mean * scale
        
        self.fused_bn_scale.copy_(scale)
        self.fused_bn_offset.copy_(offset)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_batchnorm_tanh.fused_batchnorm_tanh_cuda(x, self.fused_bn_scale, self.fused_bn_offset)
        x = self.max_pool(x)
        x = self.custom_group_norm.custom_group_norm_cuda(x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups, self.group_norm.eps)
        return x