import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for linear + instance norm + residual + multiplication
fused_operation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_operation_kernel(
    const float* x, const float* y, const float* weight, const float* bias,
    float* running_mean, float* running_var, float* output,
    int batch_size, int in_features, int out_features,
    float eps, float momentum, bool training) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;
    
    int batch_idx = idx / out_features;
    int feature_idx = idx % out_features;
    
    // Linear transformation
    float linear_result = bias[feature_idx];
    for (int i = 0; i < in_features; i++) {
        linear_result += x[batch_idx * in_features + i] * weight[feature_idx * in_features + i];
    }
    
    // Instance normalization statistics
    float mean = 0.0f;
    float var = 0.0f;
    int count = 1;  // For 2D instance norm, we treat each element as a separate instance
    
    float val = linear_result;
    mean += val;
    var += val * val;
    
    mean /= count;
    var = var / count - mean * mean;
    
    // Normalize
    float normalized = (linear_result - mean) / sqrtf(var + eps);
    
    // Residual connection and multiplication
    float y_val = y[batch_idx * out_features + feature_idx];
    output[idx] = (normalized + y_val) * y_val;
    
    // Update running statistics during training
    if (training) {
        atomicAdd(&running_mean[feature_idx], momentum * (mean - running_mean[feature_idx]));
        atomicAdd(&running_var[feature_idx], momentum * (var - running_var[feature_idx]));
    }
}

torch::Tensor fused_operation_cuda(
    torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    float eps, float momentum, bool training) {
    
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, x.options());
    
    const int block_size = 256;
    const int num_elements = batch_size * out_features;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    fused_operation_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_features, out_features, eps, momentum, training);
    
    return output;
}
"""

fused_operation_cpp_source = """
torch::Tensor fused_operation_cuda(
    torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    float eps, float momentum, bool training);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_operation_cpp_source,
    cuda_sources=fused_operation_source,
    functions=["fused_operation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        
        # Linear layer parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Instance normalization running statistics
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_operation

    def forward(self, x, y):
        return self.fused_op.fused_operation_cuda(
            x, y, self.weight, self.bias, 
            self.running_mean, self.running_var,
            self.eps, self.momentum, self.training
        )