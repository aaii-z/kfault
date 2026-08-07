import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul, swish, bias, and group norm
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Define block size for reduction operations
#define BLOCK_SIZE 256

// Fused kernel: Matrix multiplication (GEMM) + Swish + Bias
// Writes output to a temporary buffer for GroupNorm
template <typename scalar_t>
__global__ void fused_matmul_swish_bias_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias_param,
    scalar_t* __restrict__ output,
    int batch_size, int in_features, int out_features) {
    
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && out_idx < out_features) {
        // Compute dot product
        scalar_t sum = 0.0f;
        int input_offset = batch_idx * in_features;
        int weight_offset = out_idx * in_features;
        
        // Loop over in_features
        for (int k = 0; k < in_features; ++k) {
            sum += input[input_offset + k] * weight[weight_offset + k];
        }
        
        // Swish Activation: x * sigmoid(x)
        scalar_t sig = 1.0f / (1.0f + expf(-sum));
        scalar_t swish_out = sum * sig;
        
        // Add Bias
        scalar_t biased = swish_out + bias_param[out_idx];
        
        output[batch_idx * out_features + out_idx] = biased;
    }
}

// Separate kernel for GroupNorm
template <typename scalar_t>
__global__ void group_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ y,
    int batch_size, int out_features, int num_groups, int group_size) {
    
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || group_idx >= num_groups) return;
    
    // 1. Compute mean and variance for this (batch, group)
    __shared__ scalar_t s_sum[BLOCK_SIZE];
    __shared__ scalar_t s_sum_sq[BLOCK_SIZE];
    
    scalar_t local_sum = 0.0f;
    scalar_t local_sum_sq = 0.0f;
    
    int start_idx = batch_idx * out_features + group_idx * group_size;
    
    // Stride loop to compute local sums
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        scalar_t val = x[start_idx + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Reduction within block
    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    scalar_t mean = s_sum[0] / group_size;
    scalar_t var = s_sum_sq[0] / group_size - mean * mean;
    scalar_t inv_std = rsqrtf(var + 1e-5f);
    
    // 2. Normalize and apply affine transform
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int idx = start_idx + i;
        int channel_idx = group_idx * group_size + i;
        scalar_t normalized = (x[idx] - mean) * inv_std;
        y[idx] = normalized * gamma[channel_idx] + beta[channel_idx];
    }
}

// Wrapper function
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_param,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // Launch GEMM + Swish + Bias kernel
    const int block_size_gemm = 32; 
    const int num_blocks_x = (out_features + block_size_gemm - 1) / block_size_gemm;
    const int num_blocks_y = batch_size;
    
    dim3 grid_gemm(num_blocks_x, num_blocks_y);
    dim3 block_gemm(block_size_gemm, 1);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_matmul_swish_bias", ([&] {
        fused_matmul_swish_bias_kernel<scalar_t><<<grid_gemm, block_gemm>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_param.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_features, out_features
        );
    }));
    
    // Launch GroupNorm kernel
    int group_size = out_features / num_groups;
    const int block_size_norm = BLOCK_SIZE;
    dim3 grid_norm(batch_size, num_groups);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "group_norm", ([&] {
        group_norm_kernel<scalar_t><<<grid_norm, block_size_norm>>>(
            output.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, out_features, num_groups, group_size
        );
    }));
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias_param, torch::Tensor gamma, torch::Tensor beta, int num_groups);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    CUDA_sources=fused_ops_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        # We keep the parameters as nn.Parameter so they are trainable and handled by PyTorch optimizers
        # We use a Linear layer just to initialize weights easily, but we will use our custom op for forward
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        
        # We need to ensure the weights are contiguous for the custom kernel
        # The custom kernel expects weight shape (out_features, in_features)
        # nn.Linear stores weight as (out_features, in_features) by default, which matches.
        
        self.fused_op = fused_ops
        self.num_groups = num_groups

    def forward(self, x):
        # Extract weights and biases from the PyTorch modules
        # We use the weights from self.matmul and self.group_norm
        # Note: self.matmul.weight is (out_features, in_features)
        # self.group_norm.weight and self.group_norm.bias are the gamma and beta
        
        return self.fused_op.fused_op_cuda(
            x, 
            self.matmul.weight, 
            self.bias, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.num_groups
        )