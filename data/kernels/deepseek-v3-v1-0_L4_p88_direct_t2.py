import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM + GroupNorm + Swish + Multiply + Swish CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_gemm_groupnorm_swish_multiply_swish_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    const float* __restrict__ multiply_weight, 
    float* __restrict__ output,
    int batch_size, int in_features, int out_features, int num_groups) {
    
    extern __shared__ float shared_mem[];
    float* group_shared = shared_mem;
    
    int group_size = out_features / num_groups;
    int group_id = blockIdx.y;
    int sample_id = blockIdx.x;
    int local_idx = threadIdx.x;
    int global_idx = group_id * group_size + local_idx;
    
    if (global_idx >= out_features) return;
    
    // Compute GEMM for this thread's output element
    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
        float input_val = input[sample_id * in_features + i];
        float weight_val = weight[global_idx * in_features + i];
        sum += input_val * weight_val;
    }
    
    if (bias != nullptr) {
        sum += bias[global_idx];
    }
    
    // Store in shared memory for GroupNorm reduction
    group_shared[local_idx] = sum;
    __syncthreads();
    
    // GroupNorm: compute mean and variance for this group
    float group_sum = 0.0f;
    float group_sq_sum = 0.0f;
    
    for (int i = 0; i < group_size; i++) {
        if (i < group_size) {
            float val = group_shared[i];
            group_sum += val;
            group_sq_sum += val * val;
        }
    }
    
    float mean = group_sum / group_size;
    float variance = (group_sq_sum / group_size) - (mean * mean);
    float inv_std = 1.0f / sqrtf(variance + 1e-5f);
    
    // Apply GroupNorm
    float normalized = (sum - mean) * inv_std;
    
    // Apply first Swish
    float swished = normalized * sigmoid(normalized);
    
    // Apply multiply weight
    float multiplied = swished * multiply_weight[global_idx];
    
    // Apply second Swish
    float final_output = multiplied * sigmoid(multiplied);
    
    output[sample_id * out_features + global_idx] = final_output;
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor multiply_weight, int num_groups) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    int group_size = out_features / num_groups;
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 grid(batch_size, num_groups);
    int shared_mem_size = group_size * sizeof(float);
    
    fused_gemm_groupnorm_swish_multiply_swish_kernel<<<grid, group_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        multiply_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_features, out_features, num_groups);
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_operation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor multiply_weight, int num_groups);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Store weight and bias as parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_operation_cuda(
            x, self.weight, self.bias, self.multiply_weight, self.num_groups
        )