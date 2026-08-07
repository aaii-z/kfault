import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM + BatchNorm + Scale + Softmax CUDA kernel
fused_gemm_bn_scale_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_bn_scale_softmax_kernel(
    float* output, const float* input, 
    const float* weight, const float* bias,
    const float* running_mean, const float* running_var,
    const float* scale, float eps, int batch_size, int out_features) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < batch_size && col < out_features) {
        int idx = row * out_features + col;
        
        // Batch normalization: (input - mean) / sqrt(var + eps)
        float bn_val = (input[idx] - running_mean[col]) * rsqrtf(running_var[col] + eps);
        
        // Add bias and apply scale
        float scaled_val = (bn_val + bias[col]) * scale[0];
        
        // Store intermediate value for softmax
        output[idx] = scaled_val;
    }
}

__global__ void softmax_kernel(float* output, const float* input, int batch_size, int out_features) {
    extern __shared__ float shared_max[];
    extern __shared__ float shared_sum[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Step 1: Find max value in row
    float max_val = -INFINITY;
    for (int i = tid; i < out_features; i += blockDim.x) {
        float val = input[row * out_features + i];
        if (val > max_val) max_val = val;
    }
    
    // Reduce max value
    shared_max[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }
    
    max_val = shared_max[0];
    __syncthreads();
    
    // Step 2: Compute exponentials and sum
    float sum_val = 0.0f;
    for (int i = tid; i < out_features; i += blockDim.x) {
        float val = input[row * out_features + i];
        float exp_val = expf(val - max_val);
        output[row * out_features + i] = exp_val;
        sum_val += exp_val;
    }
    
    // Reduce sum
    shared_sum[tid] = sum_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    sum_val = shared_sum[0];
    __syncthreads();
    
    // Step 3: Normalize
    for (int i = tid; i < out_features; i += blockDim.x) {
        output[row * out_features + i] /= sum_val;
    }
}

torch::Tensor fused_gemm_bn_scale_softmax_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor scale, float eps) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);
    CHECK_INPUT(scale);
    
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    
    // Allocate output tensor
    auto output = torch::empty({batch_size, out_features}, x.options());
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Perform GEMM: output = x * weight^T
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                x.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    cublasDestroy(handle);
    
    // Launch fused BN + Scale kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim((batch_size + block_dim.x - 1) / block_dim.x,
                  (out_features + block_dim.y - 1) / block_dim.y);
    
    fused_bn_scale_softmax_kernel<<<grid_dim, block_dim>>>(
        output.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        scale.data_ptr<float>(), eps, batch_size, out_features);
    
    // Launch softmax kernel
    int softmax_threads = 256;
    int softmax_blocks = batch_size;
    size_t shared_mem_size = softmax_threads * sizeof(float) * 2;
    
    softmax_kernel<<<softmax_blocks, softmax_threads, shared_mem_size>>>(
        output.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_features);
    
    return output;
}
"""

fused_gemm_bn_scale_softmax_cpp_source = """
torch::Tensor fused_gemm_bn_scale_softmax_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor scale, float eps);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_bn_scale_softmax",
    cpp_sources=fused_gemm_bn_scale_softmax_cpp_source,
    cuda_sources=fused_gemm_bn_scale_softmax_source,
    functions=["fused_gemm_bn_scale_softmax_cuda"],
    verbose=True,
    extra_cflags=["-lcublas"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        
        # Store parameters separately
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.running_mean = nn.Parameter(torch.empty(out_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.empty(out_features), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5.0**0.5)
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_gemm_bn_scale_softmax_cuda(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.scale, self.bn_eps
        )