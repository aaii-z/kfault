import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + Swish + Mul + Swish
fused_gn_swish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp reduction for sum
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel for fused operations
// Assumes features_per_group is 32 (warp size) for this specific architecture
// Grid: (batch_size * num_groups)
// Block: features_per_group (32)
__global__ void fused_gn_swish_mul_swish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ weight,
    int batch_size,
    int out_features,
    int num_groups) {

    int features_per_group = out_features / num_groups;
    
    // Calculate indices
    int group_instance_idx = blockIdx.x; // 0 to batch_size * num_groups - 1
    int batch_idx = group_instance_idx / num_groups;
    int group_idx = group_instance_idx % num_groups;
    int feature_offset_in_group = threadIdx.x; // 0 to features_per_group - 1

    // Bounds check (though threads should match features_per_group exactly)
    if (feature_offset_in_group >= features_per_group) return;

    int global_feature_idx = group_idx * features_per_group + feature_offset_in_group;
    int global_data_idx = batch_idx * out_features + global_feature_idx;

    // Load input
    float x_val = __ldg(&input[global_data_idx]);

    // 1. GroupNorm: Compute Mean and Var
    // We use warp reduction since features_per_group is 32 (one warp)
    float sum = x_val;
    sum = warp_reduce_sum(sum);
    
    // Broadcast sum from lane 0 to all lanes
    sum = __shfl_sync(0xffffffff, sum, 0);
    
    float mean = sum / features_per_group;

    float diff = x_val - mean;
    float sq_sum = diff * diff;
    sq_sum = warp_reduce_sum(sq_sum);
    
    // Broadcast sq_sum from lane 0 to all lanes
    sq_sum = __shfl_sync(0xffffffff, sq_sum, 0);
    
    float var = sq_sum / features_per_group;

    // Normalize
    float inv_std = rsqrtf(var + 1e-5f);
    float x_norm = diff * inv_std;

    // Apply Gamma/Beta (GroupNorm affine)
    float g = __ldg(&gamma[global_feature_idx]);
    float b = __ldg(&beta[global_feature_idx]);
    float x_gn = g * x_norm + b;

    // 2. Swish: x * sigmoid(x)
    float sig1 = 1.0f / (1.0f + expf(-x_gn));
    float x_sw1 = x_gn * sig1;

    // 3. Multiply
    float w = __ldg(&weight[global_feature_idx]);
    float x_mul = x_sw1 * w;

    // 4. Swish
    float sig2 = 1.0f / (1.0f + expf(-x_mul));
    float x_sw2 = x_mul * sig2;

    // Write output
    output[global_data_idx] = x_sw2;
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor weight,
    int num_groups) {

    auto batch_size = input.size(0);
    auto out_features = input.size(1);
    auto features_per_group = out_features / num_groups;

    auto output = torch::empty_like(input);

    int blocks = batch_size * num_groups;
    int threads = features_per_group; // 32

    // Launch kernel
    fused_gn_swish_mul_swish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        weight.data_ptr<float>(),
        batch_size,
        out_features,
        num_groups
    );

    return output;
}
"""

fused_gn_swish_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor weight, int num_groups);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_gn_swish_cpp_source,
    cuda_sources=fused_gn_swish_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a GEMM followed by a fused GroupNorm, Swish, Multiply, and Swish operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.fused_ops = fused_ops
        self.num_groups = num_groups

    def forward(self, x):
        # GEMM (using optimized PyTorch Linear)
        x = self.gemm(x)
        
        # Fused GroupNorm + Swish + Mul + Swish
        x = self.fused_ops.fused_op_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.multiply_weight, 
            self.num_groups
        )
        return x