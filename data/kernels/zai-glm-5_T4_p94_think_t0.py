import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <int BLOCK_SIZE>
__global__ void fused_gemm_bias_hardtanh_mish_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ bias1, 
    const float* __restrict__ bias2, 
    float* __restrict__ C, 
    int M, int N, int K) {
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1]; // Padding to avoid bank conflicts

    float sum = 0.0f;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // Load A tile
        if (row < M && (k + tx) < K) {
            As[ty][tx] = A[row * K + k + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile
        // B is (N, K). We need B[col][k + tx]?
        // No, we need B[col_start + ty][k + tx] to fill Bs rows.
        // Bs[ty] should contain B[col_start + ty].
        // Thread (ty, tx) loads B[bx * BLOCK_SIZE + ty][k + tx]
        int b_row = bx * BLOCK_SIZE + ty;
        if (b_row < N && (k + tx) < K) {
            Bs[ty][tx] = B[b_row * K + k + tx];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute
        // Thread (ty, tx) computes C[row][col].
        // Needs A[row][...] -> As[ty][...]
        // Needs B[col][...] -> Bs[tx][...] (since Bs[tx] corresponds to col_start + tx)
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[ty][i] * Bs[tx][i];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        sum += bias1[col] + bias2[col];
        sum = fmaxf(-1.0f, fminf(1.0f, sum));
        sum = sum * tanhf(logf(1.0f + expf(sum)));
        C[row * N + col] = sum;
    }
}

__global__ void group_norm_kernel(
    float* __restrict__ x, 
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size, int num_channels, int num_groups) {
    
    int batch_idx = blockIdx.x;
    int group_idx = threadIdx.x;
    
    int channels_per_group = num_channels / num_groups;
    int start_c = group_idx * channels_per_group;
    
    // Compute Mean
    float mean = 0.0f;
    for (int i = 0; i < channels_per_group; ++i) {
        mean += x[batch_idx * num_channels + start_c + i];
    }
    mean /= channels_per_group;
    
    // Compute Var
    float var = 0.0f;
    for (int i = 0; i < channels_per_group; ++i) {
        float diff = x[batch_idx * num_channels + start_c + i] - mean;
        var += diff * diff;
    }
    var /= channels_per_group;
    
    // Normalize and Apply Affine
    float inv_std = rsqrtf(var + 1e-5f);
    for (int i = 0; i < channels_per_group; ++i) {
        int idx = batch_idx * num_channels + start_c + i;
        float val = (x[idx] - mean) * inv_std;
        val = val * weight[start_c + i] + bias[start_c + i];
        x[idx] = val;
    }
}

// Helper functions to launch kernels
void fused_gemm_bias_hardtanh_mish_cuda(
    torch::Tensor A, torch::Tensor B, 
    torch::Tensor bias1, torch::Tensor bias2, torch::Tensor C, 
    int M, int N, int K) {
    
    const int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    fused_gemm_bias_hardtanh_mish_kernel<BLOCK_SIZE><<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), 
        bias1.data_ptr<float>(), bias2.data_ptr<float>(), 
        C.data_ptr<float>(), M, N, K);
}

void group_norm_cuda(
    torch::Tensor x, 
    torch::Tensor weight, torch::Tensor bias, 
    int batch_size, int num_channels, int num_groups) {
    
    group_norm_kernel<<<batch_size, num_groups>>>(
        x.data_ptr<float>(), 
        weight.data_ptr<float>(), bias.data_ptr<float>(), 
        batch_size, num_channels, num_groups);
}
"""

cpp_source = """
void fused_gemm_bias_hardtanh_mish_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias1, torch::Tensor bias2, torch::Tensor C, int M, int N, int K);
void group_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int batch_size, int num_channels, int num_groups);
"""

# Compile
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_gemm_bias_hardtanh_mish_cuda", "group_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Parameters
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        
        self.fused_ops = fused_ops

    def forward(self, x):
        batch_size = x.shape[0]
        # Allocate output tensor
        out = torch.empty(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        # Fused GEMM + Bias + Hardtanh + Mish
        self.fused_ops.fused_gemm_bias_hardtanh_mish_cuda(
            x, self.gemm.weight, self.gemm.bias, self.bias, out, 
            batch_size, self.out_features, self.in_features
        )
        
        # GroupNorm (in-place)
        self.fused_ops.group_norm_cuda(
            out, self.groupnorm.weight, self.groupnorm.bias, 
            batch_size, self.out_features, self.num_groups
        )
        
        return out