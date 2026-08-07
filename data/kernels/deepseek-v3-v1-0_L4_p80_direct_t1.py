import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for GEMM + max + mean subtraction + GELU
fused_gemm_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr float GELU_COEF_A = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

__device__ __forceinline__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__global__ void add_bias_kernel(float* output, const float* bias, int batch_size, int out_features) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        int idx = row * out_features + col;
        output[idx] += bias[col];
    }
}

__global__ void compute_max_values_kernel(const float* input, float* max_values, int batch_size, int out_features) {
    extern __shared__ float shared_max[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    float thread_max = -INFINITY;
    
    for (int i = tid; i < out_features; i += blockDim.x) {
        float val = input[row * out_features + i];
        if (val > thread_max) {
            thread_max = val;
        }
    }
    
    shared_max[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        max_values[row] = shared_max[0];
    }
}

__global__ void compute_mean_values_kernel(const float* input, float* mean_values, int batch_size, int out_features) {
    extern __shared__ float shared_sum[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    float thread_sum = 0.0f;
    
    for (int i = tid; i < out_features; i += blockDim.x) {
        thread_sum += input[row * out_features + i];
    }
    
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        mean_values[row] = shared_sum[0] / out_features;
    }
}

__global__ void fused_activation_kernel(
    float* output, const float* input, 
    int batch_size, int out_features, 
    const float* max_values, const float* mean_values) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        int idx = row * out_features + col;
        float val = input[idx];
        val = val - max_values[row] - mean_values[row];
        output[idx] = gelu_activation(val);
    }
}

torch::Tensor fused_gemm_max_mean_gelu_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int max_dim) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    auto max_values = torch::empty({batch_size}, input.options());
    auto mean_values = torch::empty({batch_size}, input.options());
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    dim3 bias_blocks((out_features + 255) / 256, batch_size);
    dim3 bias_threads(256);
    add_bias_kernel<<<bias_blocks, bias_threads>>>(
        output.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        batch_size, out_features);
    
    dim3 max_blocks(batch_size);
    dim3 max_threads(256);
    compute_max_values_kernel<<<max_blocks, max_threads, max_threads.x * sizeof(float)>>>(
        output.data_ptr<float>(), 
        max_values.data_ptr<float>(), 
        batch_size, out_features);
    
    compute_mean_values_kernel<<<max_blocks, max_threads, max_threads.x * sizeof(float)>>>(
        output.data_ptr<float>(), 
        mean_values.data_ptr<float>(), 
        batch_size, out_features);
    
    dim3 block_dim(16, 16);
    dim3 grid_dim((out_features + block_dim.x - 1) / block_dim.x,
                 (batch_size + block_dim.y - 1) / block_dim.y);
    
    fused_activation_kernel<<<grid_dim, block_dim>>>(
        output.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, out_features,
        max_values.data_ptr<float>(), 
        mean_values.data_ptr<float>());
    
    cublasDestroy(handle);
    return output;
}
"""

fused_gemm_activation_cpp_source = """
torch::Tensor fused_gemm_max_mean_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int max_dim);
"""

fused_gemm_activation = load_inline(
    name="fused_gemm_activation",
    cpp_sources=fused_gemm_activation_cpp_source,
    cuda_sources=fused_gemm_activation_source,
    functions=["fused_gemm_max_mean_gelu_cuda"],
    verbose=False,
    extra_cflags=["-lcublas"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_dim = max_dim
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_gemm_activation.fused_gemm_max_mean_gelu_cuda(
            x, self.weight, self.bias, self.max_dim
        )