import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused GEMM + Sigmoid + GEMM + LogSumExp CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

__device__ float warp_reduce_sum(float val) {
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

__device__ float warp_reduce_max(float val) {
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    return val;
}

__global__ void gemm_sigmoid_gemm_logsumexp_kernel(
    const float* input, const float* weight1, const float* bias1,
    const float* weight2, const float* bias2, float* output,
    int batch_size, int input_size, int hidden_size, int output_size) {
    
    extern __shared__ float shared_mem[];
    float* hidden_act = shared_mem;
    
    // First GEMM: input * weight1 + bias1
    for (int i = blockIdx.x; i < batch_size; i += gridDim.x) {
        // Compute first layer output with sigmoid
        for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
            float sum = bias1[j];
            for (int k = 0; k < input_size; ++k) {
                sum += input[i * input_size + k] * weight1[k * hidden_size + j];
            }
            hidden_act[j] = 1.0f / (1.0f + expf(-sum));
        }
        __syncthreads();
        
        // Second GEMM: hidden_act * weight2 + bias2
        for (int j = threadIdx.x; j < output_size; j += blockDim.x) {
            float sum = bias2[j];
            for (int k = 0; k < hidden_size; ++k) {
                sum += hidden_act[k] * weight2[k * output_size + j];
            }
            output[i * output_size + j] = sum;
        }
        __syncthreads();
    }
    
    // LogSumExp reduction per sample
    for (int i = blockIdx.x; i < batch_size; i += gridDim.x) {
        float max_val = -FLT_MAX;
        float sum_exp = 0.0f;
        
        // Find max value in the row
        for (int j = threadIdx.x; j < output_size; j += blockDim.x) {
            float val = output[i * output_size + j];
            max_val = fmaxf(max_val, val);
        }
        max_val = warp_reduce_max(max_val);
        if (threadIdx.x % 32 == 0) {
            shared_mem[threadIdx.x / 32] = max_val;
        }
        __syncthreads();
        
        if (threadIdx.x < 32) {
            max_val = shared_mem[threadIdx.x];
            for (int offset = 16; offset > 0; offset /= 2) {
                max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
            }
            if (threadIdx.x == 0) {
                shared_mem[0] = max_val;
            }
        }
        __syncthreads();
        max_val = shared_mem[0];
        
        // Compute sum of exponentials
        for (int j = threadIdx.x; j < output_size; j += blockDim.x) {
            float val = output[i * output_size + j];
            sum_exp += expf(val - max_val);
        }
        sum_exp = warp_reduce_sum(sum_exp);
        if (threadIdx.x % 32 == 0) {
            shared_mem[threadIdx.x / 32] = sum_exp;
        }
        __syncthreads();
        
        if (threadIdx.x < 32) {
            sum_exp = shared_mem[threadIdx.x];
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
            }
            if (threadIdx.x == 0) {
                output[i] = logf(sum_exp) + max_val;
            }
        }
        __syncthreads();
    }
}

torch::Tensor fused_gemm_sigmoid_gemm_logsumexp_cuda(
    torch::Tensor input, torch::Tensor weight1, torch::Tensor bias1,
    torch::Tensor weight2, torch::Tensor bias2) {
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight1.size(1);
    int output_size = weight2.size(1);
    
    auto output = torch::empty({batch_size, output_size}, input.options());
    auto final_output = torch::empty({batch_size}, input.options());
    
    dim3 grid_dim(min(batch_size, 256));
    dim3 block_dim(256);
    size_t shared_mem_size = hidden_size * sizeof(float) + 256 * sizeof(float);
    
    gemm_sigmoid_gemm_logsumexp_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(), weight1.data_ptr<float>(), bias1.data_ptr<float>(),
        weight2.data_ptr<float>(), bias2.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, input_size, hidden_size, output_size);
    
    cudaDeviceSynchronize();
    
    return final_output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_gemm_sigmoid_gemm_logsumexp_cuda(
    torch::Tensor input, torch::Tensor weight1, torch::Tensor bias1,
    torch::Tensor weight2, torch::Tensor bias2);
"""

# Compile the inline CUDA code
fused_kernel = load_inline(
    name="fused_kernel",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_sigmoid_gemm_logsumexp_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weight1 = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bias1 = nn.Parameter(torch.randn(hidden_size))
        self.weight2 = nn.Parameter(torch.randn(hidden_size, output_size))
        self.bias2 = nn.Parameter(torch.randn(output_size))
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
        bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
        bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
        nn.init.uniform_(self.bias1, -bound1, bound1)
        nn.init.uniform_(self.bias2, -bound2, bound2)
        
        self.fused_op = fused_kernel

    def forward(self, x):
        return self.fused_op.fused_gemm_sigmoid_gemm_logsumexp_cuda(
            x, self.weight1, self.bias1, self.weight2, self.bias2
        )