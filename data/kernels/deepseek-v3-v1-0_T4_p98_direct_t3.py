import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Matmul_AvgPool_GELU_Scale_Max
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr float GELU_COEF_A = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

__device__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__global__ void fused_matmul_avgpool_gelu_scale_kernel(
    const float* input, const float* weight, const float* bias,
    float* intermediate, int batch_size, int in_features, int out_features,
    int pool_kernel_size, float scale_factor) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    
    if (out_idx >= out_features || batch_idx >= batch_size) return;
    
    // Matmul + AvgPool
    float sum = 0.0f;
    int pool_start = out_idx * pool_kernel_size;
    int pool_end = min(pool_start + pool_kernel_size, in_features);
    int actual_pool_size = pool_end - pool_start;
    
    for (int i = pool_start; i < pool_end; i++) {
        float input_val = input[batch_idx * in_features + i];
        float weight_val = weight[out_idx * in_features + i];
        sum += input_val * weight_val;
    }
    
    if (bias) {
        sum += bias[out_idx];
    }
    
    float avg_pooled = sum / actual_pool_size;
    
    // GELU + Scale
    intermediate[batch_idx * out_features + out_idx] = gelu_approx(avg_pooled) * scale_factor;
}

__global__ void max_reduction_kernel(const float* input, float* output, int batch_size, int out_features) {
    extern __shared__ float shared_max[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    float thread_max = -FLT_MAX;
    for (int i = tid; i < out_features; i += blockDim.x) {
        float val = input[batch_idx * out_features + i];
        if (val > thread_max) {
            thread_max = val;
        }
    }
    
    shared_max[tid] = thread_max;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch_idx] = shared_max[0];
    }
}

torch::Tensor fused_matmul_avgpool_gelu_scale_max_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int pool_kernel_size, float scale_factor) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto intermediate = torch::empty({batch_size, out_features}, input.options());
    auto output = torch::empty({batch_size}, input.options());
    
    // First kernel: matmul + avgpool + gelu + scale
    dim3 block_dim1(256);
    dim3 grid_dim1((out_features + block_dim1.x - 1) / block_dim1.x, batch_size);
    
    fused_matmul_avgpool_gelu_scale_kernel<<<grid_dim1, block_dim1>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        intermediate.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        pool_kernel_size,
        scale_factor
    );
    
    // Second kernel: max reduction
    dim3 block_dim2(256);
    dim3 grid_dim2(batch_size);
    int shared_mem_size = block_dim2.x * sizeof(float);
    
    max_reduction_kernel<<<grid_dim2, block_dim2, shared_mem_size>>>(
        intermediate.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_avgpool_gelu_scale_max_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int pool_kernel_size, float scale_factor);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_matmul_avgpool_gelu_scale_max",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_avgpool_gelu_scale_max_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        self.fused_op = fused_operator

    def forward(self, x):
        return self.fused_op.fused_matmul_avgpool_gelu_scale_max_cuda(
            x, self.weight, self.bias, self.pool_kernel_size, self.scale_factor
        )