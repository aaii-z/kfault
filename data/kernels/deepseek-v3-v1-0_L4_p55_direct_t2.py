import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + maxpool + sum + scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_matmul_maxpool_sum_scale_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    int kernel_size, float scale_factor) {
    
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int out_feature_idx = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* current_input = input + batch_idx * in_features;
    
    // Compute matmul result for this output feature
    float result = 0.0f;
    for (int i = 0; i < in_features; i++) {
        result += current_input[i] * weight[out_feature_idx * in_features + i];
    }
    
    // Apply bias
    if (bias != nullptr) {
        result += bias[out_feature_idx];
    }
    
    // Store result in shared memory for maxpool
    shared_mem[out_feature_idx] = result;
    __syncthreads();
    
    // Max pooling with kernel_size=2
    if (out_feature_idx % 2 == 0 && out_feature_idx + 1 < out_features) {
        float max_val = fmaxf(shared_mem[out_feature_idx], shared_mem[out_feature_idx + 1]);
        shared_mem[out_feature_idx / 2] = max_val;
    }
    __syncthreads();
    
    // Sum reduction across pooled features
    if (out_feature_idx == 0) {
        float sum_val = 0.0f;
        int pooled_features = (out_features + 1) / 2;
        for (int i = 0; i < pooled_features; i++) {
            sum_val += shared_mem[i];
        }
        output[batch_idx] = sum_val * scale_factor;
    }
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, float scale_factor) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size}, input.options());
    
    dim3 grid(batch_size);
    dim3 block(out_features);
    int shared_mem_size = out_features * sizeof(float);
    
    fused_matmul_maxpool_sum_scale_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        kernel_size,
        scale_factor
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_operation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, float scale_factor);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_operation_cuda(
            x, self.weight, self.bias, self.kernel_size, self.scale_factor
        )