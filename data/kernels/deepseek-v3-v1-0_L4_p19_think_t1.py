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

__global__ void compute_statistics_kernel(
    const float* input,
    float* sum_out,
    float* sq_sum_out,
    int num_elements,
    int group_size,
    int spatial_size
) {
    extern __shared__ float shared[];
    float* sum_shared = shared;
    float* sq_sum_shared = &shared[blockDim.x];

    int tid = threadIdx.x;
    sum_shared[tid] = 0.0f;
    sq_sum_shared[tid] = 0.0f;

    for (int i = tid; i < num_elements; i += blockDim.x) {
        float val = input[i];
        sum_shared[tid] += val;
        sq_sum_shared[tid] += val * val;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
            sq_sum_shared[tid] += sq_sum_shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum_out[blockIdx.x] = sum_shared[0];
        sq_sum_out[blockIdx.x] = sq_sum_shared[0];
    }
}

__global__ void apply_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* sum_arr,
    const float* sq_sum_arr,
    float* output,
    int num_elements,
    int group_size,
    int spatial_size,
    float eps
) {
    int group_id = blockIdx.x;
    int tid = threadIdx.x;

    float sum = sum_arr[group_id];
    float sq_sum = sq_sum_arr[group_id];
    
    float mean = sum / (group_size * spatial_size);
    float variance = sq_sum / (group_size * spatial_size) - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    int group_start = group_id * group_size * spatial_size;
    int group_end = group_start + group_size * spatial_size;

    for (int i = group_start + tid; i < group_end; i += blockDim.x) {
        int channel_idx = (i / spatial_size) % group_size;
        float val = input[i];
        val = (val - mean) * inv_std;
        if (gamma && beta) {
            val = val * gamma[channel_idx] + beta[channel_idx];
        }
        output[i] = val;
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
    int spatial_size = height * width;
    int group_size = channels / num_groups;

    auto output = torch::empty_like(input);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto sum_arr = torch::zeros({batch_size * num_groups}, options);
    auto sq_sum_arr = torch::zeros({batch_size * num_groups}, options);

    dim3 stat_blocks(batch_size * num_groups);
    dim3 stat_threads(BLOCK_SIZE);
    size_t shared_mem = 2 * BLOCK_SIZE * sizeof(float);

    // Apply GELU and compute statistics
    compute_statistics_kernel<<<stat_blocks, stat_threads, shared_mem>>>(
        input.data_ptr<float>(),
        sum_arr.data_ptr<float>(),
        sq_sum_arr.data_ptr<float>(),
        input.numel(),
        group_size,
        spatial_size
    );

    // Apply normalization
    apply_norm_kernel<<<stat_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        sum_arr.data_ptr<float>(),
        sq_sum_arr.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        group_size,
        spatial_size,
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