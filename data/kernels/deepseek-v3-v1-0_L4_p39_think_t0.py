import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for scale + batch normalization
fused_scale_bn_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_scale_bn_kernel(
    const float* input,
    const float* scale,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int64_t batch_size,
    int64_t num_features,
    float eps
) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch_size * num_features) {
        int64_t feature_idx = index % num_features;
        int64_t batch_idx = index / num_features;
        
        float inv_std = 1.0f / sqrtf(running_var[feature_idx] + eps);
        float normalized = (input[index] * scale[feature_idx] - running_mean[feature_idx]) * inv_std;
        output[index] = normalized * weight[feature_idx] + bias[feature_idx];
    }
}

torch::Tensor fused_scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    auto output = torch::empty_like(input);
    int64_t batch_size = input.size(0);
    int64_t num_features = input.size(1);
    
    const int threads = 256;
    int blocks = (batch_size * num_features + threads - 1) / threads;
    
    fused_scale_bn_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        eps
    );
    
    return output;
}
"""

fused_scale_bn_cpp_source = """
torch::Tensor fused_scale_bn_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
);
"""

# Compile the inline CUDA code
fused_scale_bn = load_inline(
    name="fused_scale_bn",
    cpp_sources=fused_scale_bn_cpp_source,
    cuda_sources=fused_scale_bn_source,
    functions=["fused_scale_bn_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.eps = eps
        
        # BatchNorm parameters
        self.weight = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.momentum = momentum

    def forward(self, x):
        x = self.gemm(x)
        
        if self.training:
            # Training path - use standard operations for proper statistics updates
            x = x * self.scale
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = x * self.weight + self.bias
        else:
            # Inference path - use fused kernel
            x = fused_scale_bn.fused_scale_bn_cuda(
                x, self.scale, self.weight, self.bias,
                self.running_mean, self.running_var, self.eps
            )
        
        return x