import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM, Max, Mean Subtraction, and GELU
fused_gemm_max_sub_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU approximation function
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Tiled GEMM kernel
template <int TILE_SIZE>
__global__ void gemm_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            a_tile[threadIdx.y][threadIdx.x] = a[row * K + a_col];
        } else {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            b_tile[threadIdx.y][threadIdx.x] = b[b_row * N + col];
        } else {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

// Kernel for Bias Addition
__global__ void add_bias_kernel(float* __restrict__ data, const float* __restrict__ bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = M * N;
    if (idx < total_size) {
        int col = idx % N;
        data[idx] += bias[col];
    }
}

// Kernel for Max Reduction over dimension 1
// Output: (M, 1) tensor containing max values for each row
__global__ void reduce_max_kernel(const float* __restrict__ input, float* __restrict__ output, int M, int N) {
    extern __shared__ float s_max[];
    
    int row = blockIdx.x; // One block per row
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (row >= M) return;
    
    // 1. Max reduction
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += block_size) {
        local_max = fmaxf(local_max, input[row * N + i]);
    }
    s_max[tid] = local_max;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[row] = s_max[0];
    }
}

// Kernel for Mean, Subtraction, and GELU
// Operates on the (M, 1) tensor output from reduce_max_kernel
__global__ void final_sub_gelu_kernel(float* __restrict__ data, int M) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // 1. Compute Mean of the max values
    __shared__ float s_mean[256];
    
    float local_sum = 0.0f;
    for (int i = tid; i < M; i += block_size) {
        local_sum += data[i];
    }
    s_mean[tid] = local_sum;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mean[tid] += s_mean[tid + s];
        }
        __syncthreads();
    }
    
    float mean_val = s_mean[0] / M;
    
    // 2. Subtract Mean and apply GELU
    for (int i = tid; i < M; i += block_size) {
        float val = data[i] - mean_val;
        data[i] = gelu(val);
    }
}

// Main entry point
torch::Tensor model_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int M = x.size(0);
    int K = x.size(1);
    int N = weight.size(0); // weight is (out_features, in_features)

    // Allocate intermediate tensor for GEMM output
    auto gemm_out = torch::empty({M, N}, x.options());
    
    // 1. GEMM
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    gemm_kernel<16><<<gridDim, blockDim>>>(x.data_ptr<float>(), weight.data_ptr<float>(), gemm_out.data_ptr<float>(), M, N, K);
    
    // 2. Bias Add
    int size = M * N;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(gemm_out.data_ptr<float>(), bias.data_ptr<float>(), M, N);
    
    // 3. Max Reduction
    auto max_out = torch::empty({M, 1}, x.options());
    int reduction_threads = 256;
    reduce_max_kernel<<<M, reduction_threads, reduction_threads * sizeof(float)>>>(gemm_out.data_ptr<float>(), max_out.data_ptr<float>(), M, N);
    
    // 4. Mean, Sub, GELU
    // Since M=1024, one block is enough.
    final_sub_gelu_kernel<<<1, 256>>>(max_out.data_ptr<float>(), M);
    
    return max_out;
}
"""

fused_gemm_max_sub_gelu_cpp_source = (
    "torch::Tensor model_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_gemm_max_sub_gelu_cpp_source,
    cuda_sources=fused_gemm_max_sub_gelu_source,
    functions=["model_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_dim = max_dim
        
        # Define weights equivalent to nn.Linear
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.model_forward_cuda(x, self.weight, self.bias)