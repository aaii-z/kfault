import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + LeakyReLU + Scale
# Using Welford's algorithm for numerical stability to match PyTorch reference
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

    // Shared memory for Welford reduction
    __shared__ float s_mean[256];
    __shared__ float s_m2[256];
    __shared__ int s_count[256];
    
    // Final values broadcasted to all threads
    __shared__ float s_final_mean;
    __shared__ float s_final_inv_std;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // 1. Welford Accumulation (Thread Local)
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;

    // Iterate over all elements in the group
    for (int i = tid; i < group_size; i += stride) {
        int b = i / channels_per_group;
        int c_offset = i % channels_per_group;
        int c = group_idx * channels_per_group + c_offset;
        int idx = b * hidden_size + c;
        
        float val = static_cast<float>(x[idx]);
        count++;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        m2 += delta * delta2;
    }

    s_mean[tid] = mean;
    s_m2[tid] = m2;
    s_count[tid] = count;
    __syncthreads();

    // 2. Block Reduction (Welford Combine)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float mean_a = s_mean[tid];
            float m2_a = s_m2[tid];
            int count_a = s_count[tid];
            
            float mean_b = s_mean[tid + s];
            float m2_b = s_m2[tid + s];
            int count_b = s_count[tid + s];
            
            int count_ab = count_a + count_b;
            if (count_ab > 0) {
                float delta = mean_b - mean_a;
                float mean_ab = mean_a + delta * (static_cast<float>(count_b) / count_ab);
                float m2_ab = m2_a + m2_b + delta * delta * count_a * count_b / count_ab;
                
                s_mean[tid] = mean_ab;
                s_m2[tid] = m2_ab;
                s_count[tid] = count_ab;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float final_mean = s_mean[0];
        float final_var = s_m2[0] / s_count[0];
        s_final_mean = final_mean;
        s_final_inv_std = rsqrtf(final_var + eps);
    }
    __syncthreads();

    float mean_val = s_final_mean;
    float inv_std_val = s_final_inv_std;

    // 3. Normalize, Affine, Activate, Scale
    for (int i = tid; i < group_size; i += stride) {
        int b = i / channels_per_group;
        int c_offset = i % channels_per_group;
        int c = group_idx * channels_per_group + c_offset;
        int idx = b * hidden_size + c;

        float val = static_cast<float>(x[idx]);
        // Normalize
        val = (val - mean_val) * inv_std_val;
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
        x = self.fused_ops.fused_gn_relu_scale_cuda(
            x, self.weight, self.bias, 
            self.num_groups, self.eps, self.negative_slope
        )
        return x