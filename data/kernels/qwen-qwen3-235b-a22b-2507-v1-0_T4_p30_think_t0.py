import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm and HardTanh
group_norm_hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void group_norm_hardtanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int out_features,
    int num_groups,
    float eps,
    float hardtanh_min,
    float hardtanh_max
) {
    int group_size = out_features / num_groups;
    int sample_id = blockIdx.y;
    int group_id = blockIdx.x;
    int group_start = group_id * group_size;
    
    int tid = threadIdx.x;
    if (tid >= group_size) return;
    
    int idx = sample_id * out_features + group_start + tid;
    float x_val = input[idx];
    
    extern __shared__ float s_data[];
    s_data[tid] = x_val;
    __syncthreads();
    
    // First reduction: sum
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = 0.0f;
    if (tid == 0) {
        mean = s_data[0] / group_size;
    }
    __syncthreads();
    if (tid == 0) {
        s_data[0] = mean;
    }
    __syncthreads();
    mean = s_data[0];
    
    // Second reduction: sum of squares
    s_data[tid] = x_val * x_val;
    __syncthreads();
    
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_sq = 0.0f;
    if (tid == 0) {
        sum_sq = s_data[0];
    }
    __syncthreads();
    if (tid == 0) {
        s_data[0] = sum_sq;
    }
    __syncthreads();
    sum_sq = s_data[0];
    
    float var = sum_sq / group_size - mean * mean;
    float inv_std = 1.0f / sqrtf(var + eps);
    
    float normalized = (x_val - mean) * inv_std;
    normalized = normalized * weight[group_start + tid] + bias[group_start + tid];
    normalized = fmaxf(hardtanh_min, fminf(hardtanh_max, normalized));
    
    output[idx] = normalized;
}

torch::Tensor group_norm_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    float hardtanh_min,
    float hardtanh_max,
    int num_groups
) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 1, "Weight must be 1D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    
    int batch_size = input.size(0);
    int out_features = input.size(1);
    TORCH_CHECK(out_features % num_groups == 0, "out_features must be divisible by num_groups");
    int group_size = out_features / num_groups;
    
    TORCH_CHECK(weight.size(0) == out_features, "Weight size mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "Bias size mismatch");
    
    auto output = torch::empty_like(input);
    
    dim3 grid(num_groups, batch_size);
    dim3 block(group_size);
    size_t shared_mem = group_size * sizeof(float);
    
    group_norm_hardtanh_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        num_groups,
        eps,
        hardtanh_min,
        hardtanh_max
    );
    
    return output;
}
"""

group_norm_hardtanh_cpp_source = (
    "torch::Tensor group_norm_hardtanh_cuda("
    "    torch::Tensor input,"
    "    torch::Tensor weight,"
    "    torch::Tensor bias,"
    "    float eps,"
    "    float hardtanh_min,"
    "    float hardtanh_max,"
    "    int num_groups);"
)

# Compile the inline CUDA code
group_norm_hardtanh = load_inline(
    name="group_norm_hardtanh",
    cpp_sources=group_norm_hardtanh_cpp_source,
    cuda_sources=group_norm_hardtanh_source,
    functions=["group_norm_hardtanh_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses GroupNorm and HardTanh into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.num_groups = num_groups
        self.eps = 1e-5  # Default GroupNorm epsilon
        self.group_norm_hardtanh = group_norm_hardtanh

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm_hardtanh.group_norm_hardtanh_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias,
            self.eps,
            self.hardtanh_min,
            self.hardtanh_max,
            self.num_groups
        )
        return x