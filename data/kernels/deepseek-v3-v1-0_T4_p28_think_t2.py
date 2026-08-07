import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Custom CUDA kernel for fused InstanceNorm + Residual + Multiplication
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void instance_norm_residual_mul_kernel(
    const float* linear_output, const float* y, float* output, 
    int batch_size, int out_features, float eps) {
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float sdata[];
    float* ssum = sdata;
    float* ssqsum = &sdata[block_size];

    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int i = tid; i < out_features; i += block_size) {
        float val = linear_output[batch_idx * out_features + i];
        sum += val;
        sq_sum += val * val;
    }

    ssum[tid] = sum;
    ssqsum[tid] = sq_sum;
    __syncthreads();

    for (int s = block_size/2; s>0; s>>=1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            ssqsum[tid] += ssqsum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = ssum[0] / out_features;
        float variance = (ssqsum[0] / out_features) - (mean * mean);
        ssum[0] = mean;
        ssqsum[0] = variance;
    }
    __syncthreads();

    float mean = ssum[0];
    float inv_std = 1.0f / sqrtf(ssqsum[0] + eps);

    for (int i = tid; i < out_features; i += block_size) {
        float val = linear_output[batch_idx * out_features + i];
        float normalized_val = (val - mean) * inv_std;
        float y_val = y[batch_idx * out_features + i];
        output[batch_idx * out_features + i] = (normalized_val + y_val) * y_val;
    }
}

torch::Tensor instance_norm_residual_mul_cuda(
    torch::Tensor linear_output, torch::Tensor y, float eps) {
    
    auto batch_size = linear_output.size(0);
    auto out_features = linear_output.size(1);
    auto output = torch::zeros_like(linear_output);

    int block_size = 256;
    int shared_mem_size = 2 * block_size * sizeof(float);

    instance_norm_residual_mul_kernel<<<batch_size, block_size, shared_mem_size>>>(
        linear_output.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        eps
    );

    return output;
}
"""

cpp_source = "torch::Tensor instance_norm_residual_mul_cuda(torch::Tensor linear_output, torch::Tensor y, float eps);"

instance_norm_residual_mul = load_inline(
    name="instance_norm_residual_mul",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["instance_norm_residual_mul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, y):
        linear_output = torch.mm(x, self.weight.t()) + self.bias
        return instance_norm_residual_mul.instance_norm_residual_mul_cuda(linear_output, y, self.eps)