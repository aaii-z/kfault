import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM+BiasAdd+Hardtanh+Mish CUDA kernel
fused_gemm_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define MISH_THRESHOLD 20.0f

__device__ float mish_activation(float x) {
    if (x > MISH_THRESHOLD) {
        return x;
    } else if (x < -MISH_THRESHOLD) {
        return 0.0f;
    }
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_gemm_activation_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    float min_val, float max_val) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[k * out_features + col];
        }
        
        // Add bias
        sum += bias[col];
        
        // Hardtanh
        sum = fminf(fmaxf(sum, min_val), max_val);
        
        // Mish activation
        sum = mish_activation(sum);
        
        output[row * out_features + col] = sum;
    }
}

torch::Tensor fused_gemm_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float min_val, float max_val) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(1);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );
    
    fused_gemm_activation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        min_val,
        max_val
    );
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_gemm_activation_cpp_source = """
torch::Tensor fused_gemm_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float min_val, float max_val);
"""

# Define the GroupNorm CUDA kernel
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void group_norm_kernel(
    const float* input, float* output, int batch_size, int channels,
    int groups, int group_size, const float* gamma, const float* beta,
    float epsilon) {
    
    int group_idx = blockIdx.x;
    int sample_idx = blockIdx.y;
    int local_idx = threadIdx.x;
    
    if (group_idx >= groups || sample_idx >= batch_size) return;
    
    extern __shared__ float shared_data[];
    float* shared_mean = shared_data;
    float* shared_variance = shared_data + 1;
    
    // Compute mean for this group
    float sum = 0.0f;
    int start = sample_idx * channels + group_idx * group_size;
    
    for (int i = local_idx; i < group_size; i += blockDim.x) {
        int idx = start + i;
        if (idx < batch_size * channels) {
            sum += input[idx];
        }
    }
    
    // Reduce sum across threads using warp shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (local_idx % 32 == 0) {
        shared_data[local_idx / 32] = sum;
    }
    
    __syncthreads();
    
    if (local_idx == 0) {
        float total_sum = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; i++) {
            total_sum += shared_data[i];
        }
        float mean = total_sum / group_size;
        *shared_mean = mean;
    }
    
    __syncthreads();
    
    // Compute variance
    float var_sum = 0.0f;
    float mean = *shared_mean;
    
    for (int i = local_idx; i < group_size; i += blockDim.x) {
        int idx = start + i;
        if (idx < batch_size * channels) {
            float diff = input[idx] - mean;
            var_sum += diff * diff;
        }
    }
    
    // Reduce variance sum across threads using warp shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    
    if (local_idx % 32 == 0) {
        shared_data[local_idx / 32] = var_sum;
    }
    
    __syncthreads();
    
    if (local_idx == 0) {
        float total_var_sum = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; i++) {
            total_var_sum += shared_data[i];
        }
        float variance = total_var_sum / group_size + epsilon;
        *shared_variance = rsqrtf(variance);
    }
    
    __syncthreads();
    
    // Normalize and apply affine transformation
    float inv_std = *shared_variance;
    
    for (int i = local_idx; i < group_size; i += blockDim.x) {
        int idx = start + i;
        if (idx < batch_size * channels) {
            int param_idx = group_idx * group_size + i;
            output[idx] = (input[idx] - mean) * inv_std * gamma[param_idx] + beta[param_idx];
        }
    }
}

torch::Tensor group_norm_cuda(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor beta,
    int groups, float epsilon) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int group_size = channels / groups;
    
    auto output = torch::empty_like(input);
    
    dim3 grid_size(groups, batch_size);
    int block_size = 256;
    int shared_mem_size = ((block_size + 31) / 32) * sizeof(float) + 2 * sizeof(float);
    
    group_norm_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        groups,
        group_size,
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        epsilon
    );
    
    cudaDeviceSynchronize();
    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_cuda(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor beta,
    int groups, float epsilon);
"""

# Compile the inline CUDA code
fused_gemm_activation = load_inline(
    name="fused_gemm_activation",
    cpp_sources=fused_gemm_activation_cpp_source,
    cuda_sources=fused_gemm_activation_source,
    functions=["fused_gemm_activation_cuda"],
    verbose=False
)

group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # GroupNorm parameters
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        
        # Hardtanh parameters
        self.min_val = -1.0
        self.max_val = 1.0
        self.epsilon = 1e-5
        
        # Load custom CUDA operators
        self.fused_gemm_activation = fused_gemm_activation
        self.group_norm_op = group_norm

    def forward(self, x):
        # Fused GEMM + BiasAdd + Hardtanh + Mish
        x = self.fused_gemm_activation.fused_gemm_activation_cuda(
            x, self.weight, self.bias, self.min_val, self.max_val
        )
        
        # GroupNorm
        x = self.group_norm_op.group_norm_cuda(
            x, self.gamma, self.beta, self.num_groups, self.epsilon
        )
        
        return x