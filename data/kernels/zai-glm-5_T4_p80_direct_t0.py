import torch
import torch.nn as nn
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

// Fused kernel for Max, Mean Subtraction, and GELU
__global__ void fused_max_sub_gelu_kernel(float* __restrict__ x, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (row >= M) return;

    float* row_data = x + row * N;

    // 1. Max reduction
    __shared__ float s_max[256];
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += stride) {
        local_max = fmaxf(local_max, row_data[i]);
    }
    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = s_max[0];

    // 2. Mean reduction
    __shared__ float s_mean[256];
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        local_sum += row_data[i];
    }
    s_mean[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mean[tid] += s_mean[tid + s];
        }
        __syncthreads();
    }
    float mean_val = s_mean[0] / N;

    // 3. Subtraction and GELU
    for (int i = tid; i < N; i += stride) {
        float val = row_data[i] - mean_val;
        row_data[i] = gelu(val);
    }
}

// Helper function to handle GEMM + Bias
void gemm_bias_cuda(const float* a, const float* b, const float* bias, float* c, int M, int N, int K) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    gemm_kernel<16><<<gridDim, blockDim>>>(a, b, c, M, N, K);

    // Add bias
    // Simple vectorized add for bias
    int size = M * N;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    // Using a simple kernel for bias add would be cleaner but let's inline or use thrust?
    // For simplicity and avoiding extra kernel launch overhead in this single source string,
    // we can write a bias kernel or just do it in the fused kernel if we restructure.
    // However, the fused kernel works row-wise.
    // Let's add a bias kernel.
    
    // Actually, let's fuse bias into the GEMM kernel or add a separate small kernel.
    // Given the constraints, a separate kernel for bias is safest to ensure correctness.
    // But to minimize launches, let's modify the GEMM kernel to add bias.
    // Re-launching GEMM kernel definition with bias is verbose.
    // Let's just add a bias add kernel.
}

__global__ void add_bias_kernel(float* data, const float* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        data[idx] += bias[col];
    }
}

// Main entry point
torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int M = x.size(0);
    int K = x.size(1);
    int N = weight.size(0); // weight is (out_features, in_features)

    auto output = torch::empty({M, N}, x.options());

    // 1. GEMM
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    gemm_kernel<16><<<gridDim, blockDim>>>(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), M, N, K);

    // 2. Add Bias
    int size = M * N;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(output.data_ptr<float>(), bias.data_ptr<float>(), M, N);

    // 3. Fused Max, Sub, GELU
    // We need to handle the max_dim. The python code says max_dim=1.
    // This means reducing over the feature dimension (N).
    // The fused kernel assumes reduction over the last dimension.
    
    // Wait, the original code:
    // x = torch.max(x, dim=self.max_dim, keepdim=True).values
    // If max_dim=1, it reduces over columns. Result shape (batch_size, 1).
    // Then x = x - x.mean(dim=1, keepdim=True).
    // This broadcasts.
    // The fused kernel I wrote reduces over the last dimension and modifies in place.
    // This matches the logic if max_dim=1.
    
    // However, the logic in the fused kernel:
    // row_data[i] - mean_val.
    // The original code does: (max_val) - mean(max_val).
    // Wait, looking at the original code again:
    // x = self.gemm(x) -> (M, N)
    // x = torch.max(x, dim=1, keepdim=True).values -> (M, 1)
    // x = x - x.mean(dim=1, keepdim=True) -> (M, 1) - scalar -> (M, 1)
    // x = gelu(x) -> (M, 1)
    // Output is (M, 1).
    
    // My previous analysis was slightly off.
    // The output of the model is (batch_size, 1).
    // The fused kernel I wrote produces (M, N) output.
    // I need to correct the kernel to match the logic.
    
    // Correct logic:
    // 1. Compute GEMM + Bias -> Temp (M, N)
    // 2. Reduce Max over dim 1 -> Maxs (M, 1)
    // 3. Compute Mean of Maxs -> Scalar (or (1, 1))
    // 4. Subtract Mean from Maxs -> Result (M, 1)
    // 5. GELU -> Result (M, 1)
    
    // This requires a different kernel strategy.
    // We can fuse GEMM + Bias + Max reduction.
    // Then a second small kernel for Mean + Sub + GELU.
    
    // Let's re-write.
}

// Kernel for GEMM + Bias + Max Reduction along columns
__global__ void gemm_bias_max_kernel(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ bias, float* __restrict__ max_vals, int M, int N, int K) {
    int row = blockIdx.y * 16;
    int col_block = blockIdx.x;
    
    // This kernel is getting complex to tile perfectly in one go without shared memory conflicts for reduction.
    // Alternative: Standard GEMM, then a reduction kernel.
    
    // Given the time, let's stick to the sequence of optimized kernels.
    // 1. GEMM (Tiled)
    // 2. Bias Add (Vectorized)
    // 3. Fused Max, Mean, Sub, GELU (Reduction)
}

// Let's refine the fused kernel for the specific reduction case.
// Input: (M, N) tensor. Output: (M, 1) tensor.
// Logic:
// val = max(row)
// val = val - mean(maxs)
// val = gelu(val)

__global__ void reduce_max_sub_gelu_kernel(const float* __restrict__ input, float* __restrict__ output, int M, int N) {
    extern __shared__ float s_data[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (row >= M) return;
    
    // 1. Max reduction
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += block_size) {
        local_max = fmaxf(local_max, input[row * N + i]);
    }
    s_data[tid] = local_max;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }
    
    // Store max to global to compute global mean in a separate step?
    // Or can we do it here? We need the mean of all maxes.
    // We can't compute mean of all maxes in a row-block kernel easily without a second pass.
    // So, first pass: Compute Maxs.
    // Second pass: Compute Mean, Sub, GELU.
    
    if (tid == 0) {
        output[row] = s_data[0]; // output is (M, 1)
    }
}

__global__ void final_sub_gelu_kernel(float* __restrict__ data, int M) {
    // data is (M, 1)
    // Compute mean
    int tid = threadIdx.x;
    __shared__ float s_mean[256];
    
    float local_sum = 0.0f;
    for (int i = tid; i < M; i += blockDim.x) {
        local_sum += data[i];
    }
    s_mean[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mean[tid] += s_mean[tid + s];
        }
        __syncthreads();
    }
    
    float mean_val = s_mean[0] / M;
    
    // Subtract and GELU
    for (int i = tid; i < M; i += blockDim.x) {
        float val = data[i] - mean_val;
        data[i] = gelu(val);
    }
}

// Final wrapper
torch::Tensor model_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int M = x.size(0);
    int K = x.size(1);
    int N = weight.size(0); // (out_features, in_features)

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
    reduce_max_sub_gelu_kernel<<<M, reduction_threads, reduction_threads * sizeof(float)>>>(gemm_out.data_ptr<float>(), max_out.data_ptr<float>(), M, N);
    
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

import math