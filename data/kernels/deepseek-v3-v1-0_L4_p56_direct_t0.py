import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + sigmoid + sum
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

__global__ void fused_matmul_sigmoid_sum_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output, 
    int batch_size, 
    int input_size, 
    int hidden_size) {
    
    extern __shared__ float shared_mem[];
    float* thread_sums = shared_mem;
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= batch_size) return;
    
    // Each thread handles multiple elements
    float thread_sum = 0.0f;
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        float dot_product = 0.0f;
        for (int k = 0; k < input_size; k++) {
            dot_product += input[row * input_size + k] * weight[col * input_size + k];
        }
        
        if (bias) {
            dot_product += bias[col];
        }
        
        // Sigmoid activation
        float activated = 1.0f / (1.0f + expf(-dot_product));
        thread_sum += activated;
    }
    
    thread_sums[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            thread_sums[tid] += thread_sums[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[row] = thread_sums[0];
    }
}

torch::Tensor fused_matmul_sigmoid_sum_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);
    
    auto output = torch::empty({batch_size, 1}, input.options());
    
    dim3 blocks(batch_size);
    dim3 threads(256);
    size_t shared_mem_size = threads.x * sizeof(float);
    
    fused_matmul_sigmoid_sum_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_matmul_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
fused_kernel = load_inline(
    name="fused_kernel",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_matmul_sigmoid_sum_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and bias manually
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5.0**0.5)
        nn.init.zeros_(self.bias)
        
        self.fused_op = fused_kernel

    def forward(self, x):
        return self.fused_op.fused_matmul_sigmoid_sum_cuda(x, self.weight, self.bias)