import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused BatchNorm + Tanh
fused_bn_tanh_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bn_tanh_kernel(
    float* input,
    const float* mean,
    const float* var,
    const float* weight,
    const float* bias,
    float eps,
    int num_features,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int c = (idx / (num_elements / num_features)) % num_features;
    
    float inv_std = 1.0f / sqrtf(var[c] + eps);
    float normalized = (input[idx] - mean[c]) * inv_std;
    float scaled = normalized * weight[c] + bias[c];
    input[idx] = tanhf(scaled);
}

torch::Tensor fused_bn_tanh_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    int num_features = input.size(1);
    int num_elements = input.numel();
    
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    fused_bn_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        eps,
        num_features,
        num_elements
    );
    
    return input;
}
"""

fused_bn_tanh_cpp_source = """
torch::Tensor fused_bn_tanh_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

# Compile the inline CUDA code
fused_bn_tanh = load_inline(
    name="fused_bn_tanh",
    cpp_sources=fused_bn_tanh_cpp_source,
    cuda_sources=fused_bn_tanh_source,
    functions=["fused_bn_tanh_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_bn_tanh.fused_bn_tanh_cuda(
            x,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.eps
        )
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x