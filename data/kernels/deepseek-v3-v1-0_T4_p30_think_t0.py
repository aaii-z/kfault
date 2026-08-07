import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm and HardTanh
group_norm_hardtanh_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>

#define CUDA_NUM_THREADS 256

inline int get_num_threads(int n) {
    return (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void group_norm_hardtanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int num_channels,
    int num_groups,
    int channels_per_group,
    float eps,
    float min_val,
    float max_val
) {
    int group_id = blockIdx.x;
    int batch_id = blockIdx.y;
    
    extern __shared__ float shared_data[];
    float* means = shared_data;
    float* variances = &shared_data[num_groups];
    
    int group_start = group_id * channels_per_group;
    int group_end = group_start + channels_per_group;
    
    // Compute mean and variance for the group
    float sum = 0.0f;
    float square_sum = 0.0f;
    
    for (int c = group_start; c < group_end; ++c) {
        int idx = batch_id * num_channels + c;
        float val = input[idx];
        sum += val;
        square_sum += val * val;
    }
    
    float mean = sum / channels_per_group;
    float variance = (square_sum / channels_per_group) - (mean * mean);
    
    means[group_id] = mean;
    variances[group_id] = variance;
    __syncthreads();
    
    // Normalize and apply HardTanh
    for (int c = group_start; c < group_end; ++c) {
        int idx = batch_id * num_channels + c;
        float val = input[idx];
        float normalized = (val - means[group_id]) / sqrt(variances[group_id] + eps);
        float scaled = normalized * weight[c] + bias[c];
        float clipped = fminf(fmaxf(scaled, min_val), max_val);
        output[idx] = clipped;
    }
}

torch::Tensor group_norm_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps,
    float min_val,
    float max_val
) {
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int channels_per_group = num_channels / num_groups;
    
    auto output = torch::empty_like(input);
    
    dim3 grid_dim(num_groups, batch_size);
    size_t shared_mem_size = 2 * num_groups * sizeof(float);
    
    group_norm_hardtanh_kernel<<<grid_dim, 1, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        channels_per_group,
        eps,
        min_val,
        max_val
    );
    
    return output;
}
"""

group_norm_hardtanh_cpp_source = """
torch::Tensor group_norm_hardtanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps,
    float min_val,
    float max_val
);
"""

# Compile the inline CUDA code
group_norm_hardtanh = load_inline(
    name="group_norm_hardtanh",
    cpp_sources=group_norm_hardtanh_cpp_source,
    cuda_sources=group_norm_hardtanh_source,
    functions=["group_norm_hardtanh_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.eps = 1e-5
        
        # Initialize weight and bias for group normalization
        self.weight = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        x = self.gemm(x)
        x = group_norm_hardtanh.group_norm_hardtanh_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.num_groups, 
            self.eps, 
            self.hardtanh_min, 
            self.hardtanh_max
        )
        return x