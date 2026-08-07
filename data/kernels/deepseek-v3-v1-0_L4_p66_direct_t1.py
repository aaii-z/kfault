import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Custom CUDA kernel for fused matmul + dropout + softmax
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

__global__ void fused_matmul_dropout_softmax_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, 
    int in_features, 
    int out_features,
    float dropout_p,
    unsigned long long seed,
    bool training) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        // Thread-local random state
        curandStatePhilox4_32_10_t state;
        curand_init(seed, row * out_features + col, 0, &state);
        
        // Matrix multiplication
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum += bias[col];
        
        // Apply dropout during training
        if (training) {
            float random_val = curand_uniform(&state);
            if (random_val < dropout_p) {
                sum = 0.0f;
            } else {
                sum /= (1.0f - dropout_p);
            }
        }
        
        output[row * out_features + col] = sum;
    }
    
    __syncthreads();
    
    // Online softmax computation (thread block level)
    if (row < batch_size) {
        // First pass: find max value in the row
        __shared__ float row_max[256];
        float max_val = -INFINITY;
        
        for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
            float val = output[row * out_features + i];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        __shared__ float shared_max[32];
        if (threadIdx.x < 32) {
            shared_max[threadIdx.x] = -INFINITY;
        }
        __syncthreads();
        
        atomicMax((int*)&shared_max[threadIdx.x % 32], __float_as_int(max_val));
        __syncthreads();
        
        if (threadIdx.x == 0) {
            for (int i = 1; i < 32; i++) {
                if (shared_max[i] > shared_max[0]) {
                    shared_max[0] = shared_max[i];
                }
            }
            row_max[0] = shared_max[0];
        }
        __syncthreads();
        
        max_val = row_max[0];
        
        // Second pass: compute exponentials and sum
        float exp_sum = 0.0f;
        for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
            float val = output[row * out_features + i];
            float exp_val = expf(val - max_val);
            output[row * out_features + i] = exp_val;
            exp_sum += exp_val;
        }
        
        __shared__ float shared_sum[32];
        if (threadIdx.x < 32) {
            shared_sum[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        atomicAdd(&shared_sum[threadIdx.x % 32], exp_sum);
        __syncthreads();
        
        if (threadIdx.x == 0) {
            for (int i = 1; i < 32; i++) {
                shared_sum[0] += shared_sum[i];
            }
            row_max[0] = shared_sum[0];
        }
        __syncthreads();
        
        exp_sum = row_max[0];
        
        // Third pass: normalize
        for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
            output[row * out_features + i] /= exp_sum;
        }
    }
}

torch::Tensor fused_matmul_dropout_softmax_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    float dropout_p,
    bool training) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    unsigned long long seed = time(NULL);
    
    dim3 block(32, 8);
    dim3 grid((out_features + block.x - 1) / block.x, 
              (batch_size + block.y - 1) / block.y);
    
    fused_matmul_dropout_softmax_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        dropout_p,
        seed,
        training
    );
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_matmul_dropout_softmax_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    float dropout_p,
    bool training);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_matmul_dropout_softmax",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_dropout_softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcurand"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_matmul_dropout_softmax_cuda(
            x, self.weight, self.bias, self.dropout_p, self.training
        )