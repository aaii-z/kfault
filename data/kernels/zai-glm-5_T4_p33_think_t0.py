import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for GEMM+Scale and BatchNorm
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constants
const int BLOCK_SIZE = 32;

// Kernel 1: Fused GEMM + Bias + Scale
// Computes Y = (X * W^T + bias) * scale
// X: (M, K), W: (N, K), bias: (N), scale: (N)
// Y: (M, N)
__global__ void gemm_scale_bias_kernel(const float* __restrict__ X, 
                                       const float* __restrict__ W, 
                                       const float* __restrict__ bias, 
                                       const float* __restrict__ scale, 
                                       float* __restrict__ Y, 
                                       int M, int N, int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Coordinate of the output element
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    // Shared memory for tiles with padding to avoid bank conflicts
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    float value = 0.0f;
    
    // Loop over K dimension
    for (int k_step = 0; k_step < K; k_step += BLOCK_SIZE) {
        // Load X tile (A)
        // We need X[row, k_step + tx]
        if (row < M && (k_step + tx) < K) {
            s_A[ty][tx] = X[row * K + k_step + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }
        
        // Load W tile (B)
        // We need W[col, k_step + ty] (since we are doing X * W^T, we need row of W)
        // W is stored (N, K). We need W[col, k...]
        if (col < N && (k_step + ty) < K) {
            s_B[tx][ty] = W[col * K + k_step + ty]; // Transposed load into shared memory
        } else {
            s_B[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            value += s_A[ty][k] * s_B[tx][k];
        }
        
        __syncthreads();
    }
    
    // Add bias and scale
    if (row < M && col < N) {
        value += bias[col];
        value *= scale[col];
        Y[row * N + col] = value;
    }
}

// Kernel 2: Batch Normalization
// Training mode: compute mean/var, update running stats, normalize
// Inference mode: normalize using running stats
__global__ void batch_norm_kernel(float* __restrict__ Y, 
                                   const float* __restrict__ bn_weight, 
                                   const float* __restrict__ bn_bias, 
                                   float* __restrict__ running_mean, 
                                   float* __restrict__ running_var, 
                                   int M, int N, float eps, float momentum, bool training) {
    int feature_idx = blockIdx.x; // One block per feature
    int tid = threadIdx.x;
    
    // Shared memory for reduction and data
    // We need to store the column of data (M elements) to avoid reloading
    extern __shared__ float s_data[];
    
    // Load data to shared memory
    for (int i = tid; i < M; i += blockDim.x) {
        s_data[i] = Y[i * N + feature_idx];
    }
    __syncthreads();
    
    float mean = 0.0f;
    float var = 0.0f;
    
    if (training) {
        // Compute sum
        float sum = 0.0f;
        for (int i = tid; i < M; i += blockDim.x) {
            sum += s_data[i];
        }
        
        // Reduction for sum
        __shared__ float s_sum[256]; // Assuming block size <= 256
        s_sum[tid] = sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sum[tid] += s_sum[tid + s];
            }
            __syncthreads();
        }
        
        mean = s_sum[0] / M;
        
        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int i = tid; i < M; i += blockDim.x) {
            float diff = s_data[i] - mean;
            sum_sq += diff * diff;
        }
        
        __shared__ float s_sum_sq[256];
        s_sum_sq[tid] = sum_sq;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sum_sq[tid] += s_sum_sq[tid + s];
            }
            __syncthreads();
        }
        
        var = s_sum_sq[0] / M;
        
        // Update running stats (only thread 0)
        if (tid == 0) {
            running_mean[feature_idx] = (1.0f - momentum) * running_mean[feature_idx] + momentum * mean;
            running_var[feature_idx] = (1.0f - momentum) * running_var[feature_idx] + momentum * var;
        }
    } else {
        // Inference mode: load running stats
        mean = running_mean[feature_idx];
        var = running_var[feature_idx];
    }
    
    // Normalize
    // Y = (Y - mean) / sqrt(var + eps) * weight + bias
    float inv_std = 1.0f / sqrtf(var + eps);
    float w = bn_weight[feature_idx];
    float b = bn_bias[feature_idx];
    
    for (int i = tid; i < M; i += blockDim.x) {
        float val = (s_data[i] - mean) * inv_std;
        s_data[i] = val * w + b;
    }
    __syncthreads();
    
    // Write back
    for (int i = tid; i < M; i += blockDim.x) {
        Y[i * N + feature_idx] = s_data[i];
    }
}

// Wrapper functions
torch::Tensor gemm_scale_bias_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor bias, torch::Tensor scale) {
    int M = X.size(0);
    int K = X.size(1);
    int N = W.size(0);
    
    auto Y = torch::zeros({M, N}, X.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    gemm_scale_bias_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), bias.data_ptr<float>(), 
        scale.data_ptr<float>(), Y.data_ptr<float>(), M, N, K);
        
    return Y;
}

torch::Tensor batch_norm_cuda(torch::Tensor Y, torch::Tensor bn_weight, torch::Tensor bn_bias, 
                              torch::Tensor running_mean, torch::Tensor running_var, 
                              float eps, float momentum, bool training) {
    int M = Y.size(0);
    int N = Y.size(1);
    
    // One block per feature
    int threads = 256;
    int blocks = N;
    size_t shared_mem_size = M * sizeof(float); // For storing the column
    
    batch_norm_kernel<<<blocks, threads, shared_mem_size>>>(
        Y.data_ptr<float>(), bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        M, N, eps, momentum, training);
        
    return Y;
}
"""

cpp_source = """
torch::Tensor gemm_scale_bias_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor bias, torch::Tensor scale);
torch::Tensor batch_norm_cuda(torch::Tensor Y, torch::Tensor bn_weight, torch::Tensor bn_bias, 
                              torch::Tensor running_mean, torch::Tensor running_var, 
                              float eps, float momentum, bool training);
"""

# Load the extension
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["gemm_scale_bias_cuda", "batch_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parameters for Linear + Scale
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # Parameters for BatchNorm
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
        self.eps = eps
        self.momentum = momentum
        
        # Initialize weights similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 1. Fused GEMM + Scale
        y = custom_ops.gemm_scale_bias_cuda(x, self.weight, self.bias, self.scale)
        
        # 2. Batch Normalization
        y = custom_ops.batch_norm_cuda(y, self.bn_weight, self.bn_bias, 
                                       self.running_mean, self.running_var, 
                                       self.eps, self.momentum, 
                                       self.training)
        return y