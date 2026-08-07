import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused kernel for GELU + GroupNorm
fused_gelu_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int BLOCK_SIZE = 256;

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / 1.41421356237f));
}

__global__ void fused_gelu_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int num_groups,
    float eps
) {
    int group_size = channels / num_groups;
    int spatial_size = height * width;
    int elements_per_group = group_size * spatial_size;
    
    int group_id = blockIdx.x;
    int sample_id = group_id / num_groups;
    int group_in_sample = group_id % num_groups;
    
    extern __shared__ float shared[];
    float* sum_shared = shared;
    float* sq_sum_shared = &shared[blockDim.x];
    
    int tid = threadIdx.x;
    sum_shared[tid] = 0.0f;
    sq_sum_shared[tid] = 0.0f;
    
    int group_start = sample_id * channels * spatial_size + group_in_sample * group_size * spatial_size;
    
    // First pass: compute statistics and apply GELU
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        int channel_offset = (i / spatial_size) % group_size;
        int spatial_offset = i % spatial_size;
        int input_idx = group_start + channel_offset * spatial_size + spatial_offset;
        
        float val = input[input_idx];
        float gelu_val = gelu(val);
        
        sum_shared[tid] += gelu_val;
        sq_sum_shared[tid] += gelu_val * gelu_val;
        output[input_idx] = gelu_val;
    }
    __syncthreads();
    
    // Parallel reduction for sum and sum of squares
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
            sq_sum_shared[tid] += sq_sum_shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float mean = sum_shared[0] / elements_per_group;
        float variance = sq_sum_shared[0] / elements_per_group - mean * mean;
        float inv_std = rsqrtf(variance + eps);
        shared[0] = mean;
        shared[1] = inv_std;
    }
    __syncthreads();
    
    float mean = shared[0];
    float inv_std = shared[1];
    
    // Second pass: apply normalization
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        int channel_offset = (i / spatial_size) % group_size;
        int spatial_offset = i % spatial_size;
        int output_idx = group_start + channel_offset * spatial_size + spatial_offset;
        
        float val = output[output_idx];
        int gamma_beta_idx = group_in_sample * group_size + channel_offset;
        
        val = (val - mean) * inv_std * gamma[gamma_beta_idx] + beta[gamma_beta_idx];
        output[output_idx] = val;
    }
}

torch::Tensor fused_gelu_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::empty_like(input);
    
    int group_size = channels / num_groups;
    int num_blocks = batch_size * num_groups;
    int shared_mem_size = 2 * BLOCK_SIZE * sizeof(float) + 2 * sizeof(float);
    
    fused_gelu_norm_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        num_groups,
        eps
    );
    
    return output;
}
"""

fused_gelu_norm_cpp_source = """
torch::Tensor fused_gelu_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps
);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_gelu_norm_cpp_source,
    cuda_sources=fused_gelu_norm_source,
    functions=["fused_gelu_norm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ops.fused_gelu_norm_cuda(x, self.gamma, self.beta, self.num_groups, self.eps)
        return x