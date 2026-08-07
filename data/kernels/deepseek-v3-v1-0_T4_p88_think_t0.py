import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel for GroupNorm + Swish + Multiply + Swish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int kThreadsPerBlock = 256;
const int kMaxGridDim = 65535;

template <typename T>
__device__ T sigmoid(T x) {
    return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x));
}

template <typename T>
__global__ void group_norm_swish_multiply_swish_kernel(
    const T* input,
    const T* weight,
    const T* group_norm_weight,
    const T* group_norm_bias,
    T* output,
    const int batch_size,
    const int num_channels,
    const int num_groups,
    const T eps) {
    
    const int group_size = num_channels / num_groups;
    const int group_id = blockIdx.y;
    const int sample_id = blockIdx.z;
    const int channel_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (channel_id >= group_size) return;
    
    const int global_channel = group_id * group_size + channel_id;
    const int index = sample_id * num_channels + global_channel;
    
    // Compute mean and variance for the group
    __shared__ T shared_mean;
    __shared__ T shared_variance;
    
    if (threadIdx.x == 0) {
        T sum = 0.0;
        T sq_sum = 0.0;
        
        for (int i = 0; i < group_size; ++i) {
            const int idx = sample_id * num_channels + group_id * group_size + i;
            T val = input[idx];
            sum += val;
            sq_sum += val * val;
        }
        
        T mean = sum / group_size;
        T variance = (sq_sum / group_size) - (mean * mean);
        
        shared_mean = mean;
        shared_variance = variance;
    }
    
    __syncthreads();
    
    // Normalize
    T x = input[index];
    T normalized = (x - shared_mean) / sqrt(shared_variance + eps);
    
    // Apply GroupNorm affine transformation
    normalized = normalized * group_norm_weight[global_channel] + group_norm_bias[global_channel];
    
    // First Swish
    T swish1 = normalized * sigmoid(normalized);
    
    // Multiply with weight
    T multiplied = swish1 * weight[global_channel];
    
    // Second Swish
    T result = multiplied * sigmoid(multiplied);
    
    output[index] = result;
}

torch::Tensor fused_group_norm_swish_multiply_swish_cuda(
    torch::Tensor input,
    torch::Tensor multiply_weight,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int num_groups,
    float eps) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(multiply_weight);
    CHECK_INPUT(group_norm_weight);
    CHECK_INPUT(group_norm_bias);
    
    const int batch_size = input.size(0);
    const int num_channels = input.size(1);
    const int group_size = num_channels / num_groups;
    
    auto output = torch::empty_like(input);
    
    dim3 blocks(
        (group_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
        num_groups,
        batch_size
    );
    
    group_norm_swish_multiply_swish_kernel<float><<<blocks, kThreadsPerBlock>>>(
        input.data_ptr<float>(),
        multiply_weight.data_ptr<float>(),
        group_norm_weight.data_ptr<float>(),
        group_norm_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_channels,
        num_groups,
        eps
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_group_norm_swish_multiply_swish_cuda(
    torch::Tensor input,
    torch::Tensor multiply_weight,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int num_groups,
    float eps);
"""

# Load the custom kernel
fused_op = load_inline(
    name='fused_op',
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=['fused_group_norm_swish_multiply_swish_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.eps = 1e-5

    def forward(self, x):
        x = self.gemm(x)
        x = fused_op.fused_group_norm_swish_multiply_swish_cuda(
            x,
            self.multiply_weight,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.eps
        )
        return x