import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels
custom_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int BLOCK_SIZE = 256;
constexpr int NUM_BLOCKS = 64;

__device__ float gelu_approximation(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void fused_subtract_mean_logsumexp_gelu_kernel(
    const float* input, const float* subtract_vec,
    float* output, int batch_size, int num_features) {
    
    __shared__ float shared_mem[BLOCK_SIZE];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    float thread_sum = 0.0f;
    float thread_max = -INFINITY;

    // First pass: subtract and find max for logsumexp
    for (int i = tid; i < num_features; i += blockDim.x) {
        float val = input[batch_idx * num_features + i] - subtract_vec[i];
        thread_sum += val;
        thread_max = fmaxf(thread_max, val);
    }

    // Reduce max and sum in shared memory
    shared_mem[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    float block_max = shared_mem[0];
    __syncthreads();

    // Second pass: compute logsumexp
    thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        float val = input[batch_idx * num_features + i] - subtract_vec[i];
        thread_sum += expf(val - block_max);
    }

    shared_mem[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    float log_sum_exp = logf(shared_mem[0]) + block_max;

    // Apply GELU and store result
    if (tid == 0) {
        output[batch_idx] = gelu_approximation(log_sum_exp);
    }
}

__global__ void residual_add_kernel(
    const float* input, const float* residual,
    float* output, int batch_size, int num_features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / num_features;
    int feat_idx = idx % num_features;
    
    if (idx < batch_size * num_features) {
        output[idx] = input[batch_idx] + residual[idx];
    }
}

torch::Tensor fused_operations_cuda(
    torch::Tensor input,
    torch::Tensor subtract_vec) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(subtract_vec);
    
    int batch_size = input.size(0);
    int num_features = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    dim3 grid_dim(batch_size);
    dim3 block_dim(BLOCK_SIZE);
    
    fused_subtract_mean_logsumexp_gelu_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        subtract_vec.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features
    );
    
    return output;
}

torch::Tensor residual_add_cuda(
    torch::Tensor input,
    torch::Tensor residual) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(residual);
    
    int batch_size = residual.size(0);
    int num_features = residual.size(1);
    auto output = torch::empty_like(residual);
    
    int total_threads = batch_size * num_features;
    int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    residual_add_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        residual.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features
    );
    
    return output;
}
"""

cpp_sources = """
torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor subtract_vec);
torch::Tensor residual_add_cuda(torch::Tensor input, torch::Tensor residual);
"""

custom_ops = load_inline(
    name='custom_ops',
    cpp_sources=cpp_sources,
    cuda_sources=custom_ops_source,
    functions=['fused_operations_cuda', 'residual_add_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.clone().detach()
        x = self.gemm(x)
        x = custom_ops.fused_operations_cuda(x, self.subtract)
        x = custom_ops.residual_add_cuda(x, original_x)
        return x