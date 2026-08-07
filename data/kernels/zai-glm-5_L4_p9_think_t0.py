import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Bias + Subtract + Multiply + ReLU
fused_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void fused_gemm_epilogue_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int M, int N, int K, 
    float subtract_val, float multiply_val) {
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    // Padding to avoid shared memory bank conflicts for B_tile
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE + 1];
    
    float sum = 0.0f;
    
    // Loop over K dimension in tiles
    for (int k_step = 0; k_step < K; k_step += TILE_SIZE) {
        // Load A_tile: input is (M, K)
        // We need input[row][k_step + tx]
        if (row < M && (k_step + tx) < K) {
            A_tile[ty][tx] = input[row * K + k_step + tx];
        } else {
            A_tile[ty][tx] = 0.0f;
        }
        
        // Load B_tile: weight is (N, K)
        // We need weight[col][k_step + k] for computation.
        // In loading, thread (ty, tx) loads weight[bx*32 + ty][k_step + tx]
        // So B_tile[ty][tx] holds weight[bx*32 + ty][...]
        // Later we access B_tile[tx][k] to get weight[bx*32 + tx][...]
        
        int weight_row = bx * TILE_SIZE + ty;
        if (weight_row < N && (k_step + tx) < K) {
            B_tile[ty][tx] = weight[weight_row * K + k_step + tx];
        } else {
            B_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_tile[ty][k] * B_tile[tx][k];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        // Add bias
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Epilogue: Subtract, Multiply, ReLU
        float val = sum;
        val = val - subtract_val;
        val = val * multiply_val;
        val = fmaxf(val, 0.0f);
        
        output[row * N + col] = val;
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float subtract_val, float multiply_val) {
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    auto output = torch::zeros({M, N}, input.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    const float* bias_ptr = bias.defined() && bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;

    fused_gemm_epilogue_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        M, N, K,
        subtract_val, multiply_val
    );
    
    return output;
}
"""

fused_gemm_cpp_source = (
    "torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float subtract_val, float multiply_val);"
)

# Compile the inline CUDA code
fused_gemm_op = load_inline(
    name="fused_gemm_op",
    cpp_sources=fused_gemm_cpp_source,
    cuda_sources=fused_gemm_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        
        # Define parameters manually to mimic nn.Linear
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize weights similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_gemm_op

    def forward(self, x):
        return self.fused_op.fused_forward_cuda(x, self.weight, self.bias, self.subtract_value, self.multiply_value)