import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define GROUP_SIZE 16

__global__ void group_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int num_channels,
    float eps
) {
    int group_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    
    int group_offset = group_idx * GROUP_SIZE;
    int input_offset = batch_idx * num_channels + group_offset;
    
    __shared__ float shared_data[GROUP_SIZE];
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < GROUP_SIZE; i += blockDim.x) {
        float x_val = input[input_offset + i];
        shared_data[i] = x_val;
    }
    __syncthreads();
    
    for (int i = 0; i < GROUP_SIZE; ++i) {
        float x_val = shared_data[i];
        sum += x_val;
        sum_sq += x_val * x_val;
    }
    
    float mean = sum / GROUP_SIZE;
    float var = (sum_sq - sum * sum / GROUP_SIZE) / GROUP_SIZE;
    float inv_std = 1.0f / sqrt(var + eps);
    
    for (int i = threadIdx.x; i < GROUP_SIZE; i += blockDim.x) {
        int channel_idx = group_offset + i;
        float x_val = input[input_offset + i];
        float y = (x_val - mean) * inv_std;
        output[input_offset + i] = y * gamma[channel_idx] + beta[channel_idx];
    }
}

torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch, channels)");
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    auto output = torch::empty_like(input);
    
    dim3 grid(num_groups, batch_size);
    dim3 block(GROUP_SIZE);
    
    group_norm_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
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

group_norm_cpp = "torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps);"

min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void min_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_channels
) {
    int batch_idx = blockIdx.x;
    int input_offset = batch_idx * num_channels;
    
    float min_val = FLT_MAX;
    int tid = threadIdx.x;
    int num_elements_per_thread = num_channels / blockDim.x;
    
    for (int i = 0; i < num_elements_per_thread; ++i) {
        int idx = tid * num_elements_per_thread + i;
        float val = input[input_offset + idx];
        min_val = min(min_val, val);
    }
    
    __shared__ float shared_min[1024];
    shared_min[tid] = min_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch_idx] = shared_min[0];
    }
}

torch::Tensor min_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch, channels)");
    auto batch_size = input.size(0);
    auto num_channels = input.size(1);
    auto output = torch::empty({batch_size}, input.options());
    
    dim3 grid(batch_size);
    dim3 block(1024);
    
    min_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels
    );
    
    return output.unsqueeze(1);
}
"""

min_cpp = "torch::Tensor min_cuda(torch::Tensor input);"

group_norm_op = load_inline(
    name="group_norm_cuda",
    cpp_sources=group_norm_cpp,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=False,
    with_cuda=True
)

min_op = load_inline(
    name="min_cuda",
    cpp_sources=min_cpp,
    cuda_sources=min_source,
    functions=["min_cuda"],
    verbose=False,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm_op = group_norm_op
        self.min_op = min_op

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm_op.group_norm_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.group_norm.num_groups, 
            self.group_norm.eps
        )
        x = self.min_op.min_cuda(x)
        x = x + self.bias
        return x