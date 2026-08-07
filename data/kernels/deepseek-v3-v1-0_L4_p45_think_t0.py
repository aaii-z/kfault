import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels
custom_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define TILE_SIZE 32
#define BLOCK_SIZE 256

// Online logsumexp implementation
__global__ void online_logsumexp_kernel(const float* input, float* output, int num_rows, int num_cols) {
    extern __shared__ float shared_mem[];
    float* max_vals = shared_mem;
    float* sum_exps = &shared_mem[blockDim.x];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_max = -INFINITY;
    float local_sum_exp = 0.0f;

    for (int col = tid; col < num_cols; col += blockDim.x) {
        float val = input[row * num_cols + col];
        if (val > local_max) {
            local_max = val;
        }
    }

    max_vals[tid] = local_max;
    __syncthreads();

    // Reduce max values
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (max_vals[tid + s] > max_vals[tid]) {
                max_vals[tid] = max_vals[tid + s];
            }
        }
        __syncthreads();
    }

    float global_max = max_vals[0];
    __syncthreads();

    for (int col = tid; col < num_cols; col += blockDim.x) {
        float val = input[row * num_cols + col];
        local_sum_exp += expf(val - global_max);
    }

    sum_exps[tid] = local_sum_exp;
    __syncthreads();

    // Reduce sum of exps
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_exps[tid] += sum_exps[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = logf(sum_exps[0]) + global_max;
    }
}

// Fused GEMM + Sigmoid kernel
__global__ void gemm_sigmoid_kernel(
    const float* A, const float* B, const float* bias,
    float* C, int M, int N, int K) {
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        float result = sum + bias[col];
        C[row * N + col] = 1.0f / (1.0f + expf(-result));
    }
}

torch::Tensor fused_gemm_sigmoid_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    auto output = torch::empty({M, N}, input.options());
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_sigmoid_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );
    
    return output;
}

torch::Tensor online_logsumexp_cuda(torch::Tensor input) {
    int num_rows = input.size(0);
    int num_cols = input.size(1);
    
    auto output = torch::empty({num_rows}, input.options());
    
    int shared_mem_size = 2 * BLOCK_SIZE * sizeof(float);
    
    online_logsumexp_kernel<<<num_rows, BLOCK_SIZE, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows,
        num_cols
    );
    
    return output;
}
"""

cpp_sources = """
torch::Tensor fused_gemm_sigmoid_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor online_logsumexp_cuda(torch::Tensor input);
"""

# Load the custom operators
custom_ops = load_inline(
    name='custom_ops',
    cpp_sources=cpp_sources,
    cuda_sources=custom_ops_source,
    functions=[
        'fused_gemm_sigmoid_cuda',
        'online_logsumexp_cuda'
    ],
    verbose=True,
    extra_cflags=['-O2'],
    extra_ldflags=['-lcublas']
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1_weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.linear1_bias = nn.Parameter(torch.empty(hidden_size))
        self.linear2_weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.linear2_bias = nn.Parameter(torch.empty(output_size))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.linear1_weight, a=sqrt(5))
        nn.init.kaiming_uniform_(self.linear2_weight, a=sqrt(5))
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.linear1_weight)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.linear2_weight)
        bound1 = 1 / sqrt(fan_in1)
        bound2 = 1 / sqrt(fan_in2)
        nn.init.uniform_(self.linear1_bias, -bound1, bound1)
        nn.init.uniform_(self.linear2_bias, -bound2, bound2)

    def forward(self, x):
        # Fused GEMM + Sigmoid
        x = custom_ops.fused_gemm_sigmoid_cuda(x, self.linear1_weight, self.linear1_bias)
        
        # Second GEMM
        x = torch.nn.functional.linear(x, self.linear2_weight, self.linear2_bias)
        
        # Online LogSumExp
        x = custom_ops.online_logsumexp_cuda(x)
        return x