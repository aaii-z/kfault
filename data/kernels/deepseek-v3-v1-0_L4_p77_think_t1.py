import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Scale + BatchNorm
scale_bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_bn_kernel(
    const float* input,
    const float* running_mean,
    const float* running_var,
    const float* weight,
    const float* bias,
    float* output,
    int numel,
    int num_channels,
    float scale_factor,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int channel_idx = (idx / (numel / num_channels)) % num_channels;
        float inv_std = 1.0f / sqrtf(running_var[channel_idx] + eps);
        output[idx] = (input[idx] * scale_factor - running_mean[channel_idx]) * inv_std * weight[channel_idx] + bias[channel_idx];
    }
}

torch::Tensor scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float eps
) {
    CHECK_CUDA(input);
    CHECK_CONTIGUOUS(input);
    CHECK_CUDA(running_mean);
    CHECK_CONTIGUOUS(running_mean);
    CHECK_CUDA(running_var);
    CHECK_CONTIGUOUS(running_var);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(weight);
    CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(bias);

    auto output = torch::empty_like(input);
    int numel = input.numel();
    int num_channels = input.size(1);

    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    scale_bn_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        num_channels,
        scale_factor,
        eps
    );

    return output;
}
"""

scale_bn_cpp_source = """
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
torch::Tensor scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float eps
);
"""

scale_bn = load_inline(
    name='scale_bn',
    cpp_sources=scale_bn_cpp_source,
    cuda_sources=scale_bn_source,
    functions=['scale_bn_cuda'],
    verbose=True
)

# Custom CUDA kernel for global average pooling
global_avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_pool3d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_channels,
    int depth,
    int height,
    int width
) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    float sum = 0.0f;
    int total_elements = depth * height * width;

    for (int d = 0; d < depth; d++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = ((n * num_channels + c) * depth + d) * height * width + h * width + w;
                sum += input[idx];
            }
        }
    }
    output[n * num_channels + c] = sum / total_elements;
}

torch::Tensor global_avg_pool3d_cuda(torch::Tensor input) {
    CHECK_CUDA(input);
    CHECK_CONTIGUOUS(input);

    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    auto output = torch::empty({batch_size, num_channels, 1, 1, 1}, input.options());

    dim3 blocks(batch_size, num_channels);
    global_avg_pool3d_kernel<<<blocks, 1>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        depth,
        height,
        width
    );

    return output;
}
"""

global_avg_pool3d_cpp_source = """
torch::Tensor global_avg_pool3d_cuda(torch::Tensor input);
"""

global_avg_pool3d = load_inline(
    name='global_avg_pool3d',
    cpp_sources=global_avg_pool3d_cpp_source,
    cuda_sources=global_avg_pool3d_source,
    functions=['global_avg_pool3d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.eps = eps
        
        # Initialize batch normalization parameters
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.weight_bn = nn.Parameter(torch.ones(out_channels))
        self.bias_bn = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = scale_bn.scale_bn_cuda(
            x, 
            self.running_mean, 
            self.running_var, 
            self.weight_bn, 
            self.bias_bn, 
            self.scale_factor, 
            self.eps
        )
        x = global_avg_pool3d.global_avg_pool3d_cuda(x)
        return x