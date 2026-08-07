import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void instance_norm_kernel(const float* x, float* out, int batch_size, int features, float eps) {
    int i = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int j = tid; j < features; j += num_threads) {
        float val = x[i * features + j];
        sum += val;
        sum_sq += val * val;
    }

    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];
    shared_sum[tid] = sum;
    shared_sum_sq[tid] = sum_sq;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sum_sq[tid] += shared_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float mean = 0.0f;
    float std = 0.0f;
    if (tid == 0) {
        mean = shared_sum[0] / features;
        float var = (shared_sum_sq[0] - shared_sum[0] * shared_sum[0] / features) / features;
        std = sqrtf(var + eps);
        shared_sum[0] = mean;
        shared_sum_sq[0] = std;
    }
    __syncthreads();

    mean = shared_sum[0];
    std = shared_sum_sq[0];

    for (int j = tid; j < features; j += num_threads) {
        float val = x[i * features + j];
        out[i * features + j] = (val - mean) / std;
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor x, float eps) {
    int batch_size = x.size(0);
    int features = x.size(1);
    auto out = torch::empty_like(x);

    int block_size = 256;
    int num_blocks = batch_size;

    instance_norm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, 
        features, 
        eps
    );

    return out;
}
"""

fused_add_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_mul_kernel(const float* x, const float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (x[idx] + y[idx]) * y[idx];
    }
}

torch::Tensor fused_add_mul_cuda(torch::Tensor x, torch::Tensor y) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_mul_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        y.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );

    return out;
}
"""

cpp_source = "torch::Tensor instance_norm_cuda(torch::Tensor x, float eps); torch::Tensor fused_add_mul_cuda(torch::Tensor x, torch::Tensor y);"

instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=cpp_source.split(';')[0],
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=True,
)

fused_add_mul = load_inline(
    name="fused_add_mul",
    cpp_sources=cpp_source.split(';')[1],
    cuda_sources=fused_add_mul_source,
    functions=["fused_add_mul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = instance_norm
        self.fused_add_mul = fused_add_mul
        self.eps = eps

    def forward(self, x, y):
        x = self.bmm(x)
        x = self.instance_norm.instance_norm_cuda(x, self.eps)
        x = self.fused_add_mul.fused_add_mul_cuda(x, y)
        return x