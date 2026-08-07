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

__device__ float atomicMax(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

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
    
    extern __shared__ float shared_mem[];
    float* row_max = shared_mem;
    float* row_sum = shared_mem + 256;
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= batch_size) return;
    
    // Matrix multiplication and dropout
    for (int col = tid; col < out_features; col += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum += bias[col];
        
        // Apply dropout during training
        if (training) {
            curandStatePhilox4_32_10_t state;
            curand_init(seed, row * out_features + col, 0, &state);
            float random_val = curand_uniform(&state);
            if (random_val < dropout_p) {
                sum = -1e20f;  // Very negative value for softmax stability
            } else {
                sum /= (1.0f - dropout_p);
            }
        }
        
        output[row * out_features + col] = sum;
    }
    __syncthreads();
    
    // Find max value in the row
    float max_val = -INFINITY;
    for (int col = tid; col < out_features; col += blockDim.x) {
        float val = output[row * out_features + col];
        if (val > max_val) max_val = val;
    }
    
    max_val = blockReduceMax(max_val);
    if (tid == 0) row_max[0] = max_val;
    __syncthreads();
    
    max_val = row_max[0];
    
    // Compute exponentials and sum
    float exp_sum = 0.0f;
    for (int col = tid; col < out_features; col += blockDim.x) {
        float val = output[row * out_features + col];
        float exp_val = expf(val - max_val);
        output[row * out_features + col] = exp_val;
        exp_sum += exp_val;
    }
    
    exp_sum = blockReduceSum(exp_sum);
    if (tid == 0) row_sum[0] = exp_sum;
    __syncthreads();
    
    exp_sum = row_sum[0];
    
    // Normalize
    for (int col = tid; col < out_features; col += blockDim.x) {
        output[row * out_features + col] /= exp_sum;
    }
}

__device__ float blockReduceMax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceMax(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
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
    
    int threads = 256;
    int shared_mem_size = 2 * 256 * sizeof(float);
    
    fused_matmul_dropout_softmax_kernel<<<batch_size, threads, shared_mem_size>>>(
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