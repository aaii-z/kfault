import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + LeakyReLU + Scale
fused_gn_relu_scale_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void fused_gn_relu_scale_kernel(
    scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    int batch_size, int hidden_size, int num_groups,
    float eps, float negative_slope) {

    int group_idx = blockIdx.x;
    int channels_per_group = hidden_size / num_groups;
    int group_size = batch_size * channels_per_group;

    // Shared memory for reduction
    __shared__ float s_mean;
    __shared__ float s_var;
    __shared__ float s_sum[256]; // Assuming block size 256
    __shared__ float s_sum_sq[256];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: Compute Mean and Variance
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Iterate over all elements in the group
    // Elements are stored as (Batch, Hidden). 
    // For a group, we process channels [g * CPG, (g+1) * CPG) for all batches.
    for (int i = tid; i < group_size; i += stride) {
        int b = i / channels_per_group;
        int c_offset = i % channels_per_group;
        int c = group_idx * channels_per_group + c_offset;
        
        float val = static_cast<float>(x[b * hidden_size + c]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_sum[0] / group_size;
        float var = s_sum_sq[0] / group_size - mean * mean;
        s_mean = mean;
        s_var = var;
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = rsqrtf(s_var + eps);

    // Pass 2: Normalize, Affine, Activate, Scale
    for (int i = tid; i < group_size; i += stride) {
        int b = i / channels_per_group;
        int c_offset = i % channels_per_group;
        int c = group_idx * channels_per_group + c_offset;
        int idx = b * hidden_size + c;

        float val = static_cast<float>(x[idx]);
        // Normalize
        val = (val - mean) * inv_std;
        // Affine
        val = val * static_cast<float>(gamma[c]) + static_cast<float>(beta[c]);
        // LeakyReLU
        val = val > 0.0f ? val : val * negative_slope;
        // Scale (x + x = 2 * x)
        val = val * 2.0f;

        x[idx] = static_cast<scalar_t>(val);
    }
}

torch::Tensor fused_gn_relu_scale_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps,
    float negative_slope) {

    int batch_size = x.size(0);
    int hidden_size = x.size(1);
    
    const int threads = 256;
    const int blocks = num_groups;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_gn_relu_scale_kernel", ([&] {
        fused_gn_relu_scale_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            batch_size, hidden_size, num_groups,
            eps, negative_slope
        );
    }));

    return x;
}
"""

fused_gn_relu_scale_cpp_source = (
    "torch::Tensor fused_gn_relu_scale_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps, float negative_slope);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_gn_relu_scale_cpp_source,
    cuda_sources=fused_gn_relu_scale_source,
    functions=["fused_gn_relu_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[],
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused GroupNorm + LeakyReLU + Scale.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        
        # GroupNorm parameters
        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.eps = eps
        self.negative_slope = negative_slope
        
        # Define learnable parameters for GroupNorm
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Forward pass using fused CUDA kernel.
        """
        x = self.fc(x)
        # Apply fused kernel: GroupNorm + LeakyReLU + Scale
        # The kernel modifies x in-place for efficiency, but PyTorch tracks this correctly
        x = self.fused_ops.fused_gn_relu_scale_cuda(
            x, self.weight, self.bias, 
            self.num_groups, self.eps, self.negative_slope
        )
        return x