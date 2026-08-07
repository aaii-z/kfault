import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Linear + InstanceNorm + Residual + Multiplication
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_linear_instance_norm_residual_mul_kernel(
    const float* x, const float* weight, const float* bias, const float* y,
    float* output, int batch_size, int in_features, int out_features, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int batch_idx = idx / out_features;
    int feat_idx = idx % out_features;

    // Linear transformation
    float val = 0.0f;
    for (int i = 0; i < in_features; i++) {
        val += x[batch_idx * in_features + i] * weight[i * out_features + feat_idx];
    }
    val += bias[feat_idx];

    // Instance normalization (for 1x1 spatial dimensions)
    float mean = val;
    float variance = eps;
    float inv_std = 1.0f / sqrtf(variance);
    val = (val - mean) * inv_std;

    // Residual addition and multiplication
    float y_val = y[batch_idx * out_features + feat_idx];
    output[idx] = (val + y_val) * y_val;
}

torch::Tensor fused_operator_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor y, float eps) {
    
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, x.options());

    int threads = 256;
    int blocks = (batch_size * out_features + threads - 1) / threads;

    fused_linear_instance_norm_residual_mul_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps
    );

    return output;
}
"""

cpp_source = "torch::Tensor fused_operator_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor y, float eps);"

fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_operator_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        # Initialize linear layer parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, y):
        return fused_operator.fused_operator_cuda(x, self.weight, self.bias, y, self.eps)