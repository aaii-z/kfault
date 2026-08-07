import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Gemm + Subtract + GlobalAvgPool + LogSumExp + GELU + ResidualAdd
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr int TILE_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_gemm_subtract_avg_logsumexp_gelu_residual_kernel(
    const float* input, const float* weight, const float* bias, const float* subtract_param,
    const float* original_x, float* output, int batch_size, int in_features, int out_features, bool has_bias) {
    
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE + 1];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int input_col = t * TILE_SIZE + threadIdx.x;
        int weight_row = t * TILE_SIZE + threadIdx.y;
        
        if (row < batch_size && input_col < in_features) {
            tile_a[threadIdx.y][threadIdx.x] = input[row * in_features + input_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < out_features && weight_row < in_features) {
            tile_b[threadIdx.y][threadIdx.x] = weight[col * in_features + weight_row];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < batch_size && col < out_features) {
        float result = sum;
        
        if (has_bias) {
            result += bias[col];
        }
        
        result -= subtract_param[col];
        
        __shared__ float row_sum[BLOCK_SIZE];
        __shared__ float row_max[BLOCK_SIZE];
        
        int tid = threadIdx.x + threadIdx.y * blockDim.x;
        if (tid == 0) {
            row_sum[row] = 0.0f;
            row_max[row] = -FLT_MAX;
        }
        __syncthreads();
        
        atomicAdd(&row_sum[row], result);
        atomicMax(&row_max[row], result);
        __syncthreads();
        
        if (col == 0) {
            float avg_val = row_sum[row] / out_features;
            float max_val = row_max[row];
            float logsumexp_val = logf(expf(avg_val - max_val)) + max_val;
            float gelu_val = gelu(logsumexp_val);
            output[row] = gelu_val + original_x[row * in_features];
        }
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor subtract_param, torch::Tensor original_x, bool has_bias) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, in_features}, input.options());
    
    dim3 grid_dim((out_features + TILE_SIZE - 1) / TILE_SIZE, 
                  (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    
    fused_gemm_subtract_avg_logsumexp_gelu_residual_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        has_bias ? bias.data_ptr<float>() : nullptr,
        subtract_param.data_ptr<float>(), original_x.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features, has_bias);
    
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor subtract_param, torch::Tensor original_x, bool has_bias);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.fused_op = fused_operator

    def forward(self, x):
        original_x = x.clone().detach()
        
        x = self.fused_op.fused_operator_cuda(
            x, self.weight, 
            self.bias if self.has_bias else torch.empty(0, device=x.device),
            self.subtract, original_x, self.has_bias
        )
        
        return x