import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

group_norm_swish_source = """
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

__global__ void group_norm_swish_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int num_channels,
    float eps
) {
    const int group_size = 32;
    int b = blockIdx.y;
    int g = blockIdx.x;
    int c_start = g * group_size;
    
    __shared__ float s_input[group_size];
    __shared__ float s_gamma[group_size];
    __shared__ float s_beta[group_size];
    
    int tid = threadIdx.x;
    int c = c_start + tid;
    
    if (tid < group_size) {
        s_input[tid] = input[b * num_channels + c];
        s_gamma[tid] = gamma[c];
        s_beta[tid] = beta[c];
    }
    __syncthreads();
    
    float sum = 0.0f;
    if (tid < group_size) {
        sum = s_input[tid];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    float mean = sum / group_size;
    
    float sq_sum = 0.0f;
    if (tid < group_size) {
        float diff = s_input[tid] - mean;
        sq_sum = diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq_sum += __shfl_down_sync(0xFFFFFFFF, sq_sum, offset);
    }
    float var = sq_sum / group_size;
    float inv_std = 1.0f / sqrt(var + eps);
    
    if (tid < group_size) {
        float x_normalized = (s_input[tid] - mean) * inv_std;
        float x_affine = s_gamma[tid] * x_normalized + s_beta[tid];
        float sigmoid_x = 1.0f / (1.0f + expf(-x_affine));
        output[b * num_channels + c] = x_affine * sigmoid_x;
    }
}

torch::Tensor group_norm_swish_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    auto output = torch::empty_like(input);
    
    const int num_groups = 256;
    dim3 blocks(num_groups, batch_size);
    dim3 threads(32);
    
    group_norm_swish_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        eps
    );
    
    return output;
}
"""

group_norm_swish_cpp = "torch::Tensor group_norm_swish_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);"

multiply_swish_source = """
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

__global__ void multiply_swish_kernel(
    const float* input,
    const float* weight,
    float* output,
    int total_elements,
    int num_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int c = idx % num_channels;
        float x = input[idx] * weight[c];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid_x;
    }
}

torch::Tensor multiply_swish_cuda(torch::Tensor input, torch::Tensor weight) {
    int total_elements = input.numel();
    int num_channels = input.size(1);
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    multiply_swish_kernel<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        num_channels
    );
    
    return output;
}
"""

multiply_swish_cpp = "torch::Tensor multiply_swish_cuda(torch::Tensor input, torch::Tensor weight);"

group_norm_swish = load_inline(
    name="group_norm_swish",
    cpp_sources=group_norm_swish_cpp,
    cuda_sources=group_norm_swish_source,
    functions=["group_norm_swish_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)

multiply_swish = load_inline(
    name="multiply_swish",
    cpp_sources=multiply_swish_cpp,
    cuda_sources=multiply_swish_source,
    functions=["multiply_swish_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.group_norm_swish = group_norm_swish
        self.multiply_swish = multiply_swish

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm_swish.group_norm_swish_cuda(x, self.group_norm.weight, self.group_norm.bias, 1e-5)
        x = self.multiply_swish.multiply_swish_cuda(x, self.multiply_weight)
        return x