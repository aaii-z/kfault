import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + LeakyReLU + Scale
group_norm_leaky_relu_scale_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename T>
__global__ void group_norm_leaky_relu_scale_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    const int num_groups,
    const int group_size,
    const int num_elements,
    const float eps,
    const float negative_slope
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    const int group_id = (index / group_size) % num_groups;
    const int channel_id = index % group_size;
    
    // Compute mean and variance for the group
    __shared__ float shared_mean[1024];
    __shared__ float shared_variance[1024];
    
    float sum = 0.0f;
    float square_sum = 0.0f;
    
    const int group_start = (index / (group_size * num_groups)) * (group_size * num_groups) + group_id * group_size;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const int idx = group_start + i;
        float val = static_cast<float>(input[idx]);
        sum += val;
        square_sum += val * val;
    }
    
    // Reduce sum and square_sum within the block
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        square_sum += __shfl_down_sync(0xffffffff, square_sum, offset);
    }
    
    if (threadIdx.x == 0) {
        const float mean = sum / group_size;
        const float variance = square_sum / group_size - mean * mean;
        shared_mean[group_id] = mean;
        shared_variance[group_id] = variance;
    }
    __syncthreads();
    
    const float mean = shared_mean[group_id];
    const float variance = shared_variance[group_id];
    const float inv_std = rsqrtf(variance + eps);
    
    // Normalize and apply affine transformation
    float x = static_cast<float>(input[index]);
    float normalized = (x - mean) * inv_std;
    float scaled = normalized * static_cast<float>(weight[channel_id]) + static_cast<float>(bias[channel_id]);
    
    // Apply LeakyReLU and scale by 2
    float activated = scaled > 0 ? scaled : scaled * negative_slope;
    output[index] = static_cast<T>(activated * 2.0f);
}

torch::Tensor group_norm_leaky_relu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps,
    float negative_slope
) {
    const int num_elements = input.numel();
    const int group_size = input.size(1) / num_groups;
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "group_norm_leaky_relu_scale", [&] {
        group_norm_leaky_relu_scale_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_groups,
            group_size,
            num_elements,
            eps,
            negative_slope
        );
    });
    
    return output;
}
"""

group_norm_leaky_relu_scale_cpp_source = """
torch::Tensor group_norm_leaky_relu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps,
    float negative_slope
);
"""

# Compile the inline CUDA code
group_norm_leaky_relu_scale = load_inline(
    name="group_norm_leaky_relu_scale",
    cpp_sources=group_norm_leaky_relu_scale_cpp_source,
    cuda_sources=group_norm_leaky_relu_scale_source,
    functions=["group_norm_leaky_relu_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        
        # Initialize GroupNorm parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        x = self.fc(x)
        x = group_norm_leaky_relu_scale.group_norm_leaky_relu_scale_cuda(
            x, self.weight, self.bias, self.num_groups, self.eps, self.negative_slope
        )
        return x