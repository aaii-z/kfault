import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + LeakyReLU + Scale
group_norm_leaky_relu_scale_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void group_norm_leaky_relu_scale_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    const int batch_size,
    const int hidden_size,
    const int num_groups,
    const int group_size,
    const float eps,
    const float negative_slope
) {
    const int group_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ float shared_mean;
    __shared__ float shared_variance;

    const int group_start = (batch_id * hidden_size) + (group_id * group_size);
    float sum = 0.0f;
    float square_sum = 0.0f;

    // Compute mean and variance for the group
    for (int i = tid; i < group_size; i += blockDim.x) {
        const int idx = group_start + i;
        float val = static_cast<float>(input[idx]);
        sum += val;
        square_sum += val * val;
    }

    // Warp reduction for sum and square_sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        square_sum += __shfl_down_sync(0xffffffff, square_sum, offset);
    }

    if (tid == 0) {
        const float mean = sum / group_size;
        const float variance = square_sum / group_size - mean * mean;
        shared_mean = mean;
        shared_variance = variance;
    }
    __syncthreads();

    // Normalize, apply affine transform, LeakyReLU, and scale
    if (tid < group_size) {
        const int idx = group_start + tid;
        const int channel_id = group_id * group_size + tid;
        float x = static_cast<float>(input[idx]);
        float normalized = (x - shared_mean) * rsqrtf(shared_variance + eps);
        float scaled = normalized * static_cast<float>(weight[channel_id]) + static_cast<float>(bias[channel_id]);
        float activated = scaled > 0 ? scaled : scaled * negative_slope;
        output[idx] = static_cast<T>(activated * 2.0f);
    }
}

torch::Tensor group_norm_leaky_relu_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps,
    float negative_slope
) {
    const int batch_size = input.size(0);
    const int hidden_size = input.size(1);
    const int group_size = hidden_size / num_groups;
    auto output = torch::empty_like(input);

    dim3 blocks(num_groups, batch_size);
    const int threads = 32;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "group_norm_leaky_relu_scale", [&] {
        group_norm_leaky_relu_scale_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            hidden_size,
            num_groups,
            group_size,
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
        self.hidden_size = hidden_size
        
        # Initialize GroupNorm parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        x = self.fc(x)
        x = group_norm_leaky_relu_scale.group_norm_leaky_relu_scale_cuda(
            x, self.weight, self.bias, self.num_groups, self.eps, self.negative_slope
        )
        return x