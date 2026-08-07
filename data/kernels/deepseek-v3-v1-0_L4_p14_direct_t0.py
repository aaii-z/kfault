import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + division + sum + scaling
fused_operation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

__global__ void fused_operation_kernel(
    const float* input, 
    const float* weight, 
    float* output, 
    float scaling_factor,
    int batch_size, 
    int input_size, 
    int hidden_size) {
    
    extern __shared__ float shared_mem[];
    float* row_cache = shared_mem;
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Each block processes one batch element
    const float* current_input = input + batch_idx * input_size;
    float* current_output = output + batch_idx;
    
    // Initialize output to 0
    if (tid == 0) {
        *current_output = 0.0f;
    }
    __syncthreads();
    
    // Process in chunks to reduce shared memory usage
    const int chunk_size = min(1024, hidden_size);
    int num_chunks = (hidden_size + chunk_size - 1) / chunk_size;
    
    float thread_sum = 0.0f;
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_idx = chunk * chunk_size;
        int end_idx = min(start_idx + chunk_size, hidden_size);
        int chunk_elements = end_idx - start_idx;
        
        // Load weight row into shared memory
        for (int i = tid; i < chunk_elements; i += blockDim.x) {
            int weight_idx = start_idx * input_size + i;
            if (weight_idx < hidden_size * input_size) {
                row_cache[i] = weight[weight_idx];
            }
        }
        __syncthreads();
        
        // Compute dot product for this chunk
        float chunk_sum = 0.0f;
        for (int i = 0; i < chunk_elements; i++) {
            int input_idx = start_idx + i;
            if (input_idx < input_size) {
                chunk_sum += current_input[input_idx] * row_cache[i];
            }
        }
        
        thread_sum += chunk_sum;
        __syncthreads();
    }
    
    // Reduce thread sums to get final result
    __shared__ float block_sum[256];
    block_sum[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_sum[tid] += block_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Final result with scaling
    if (tid == 0) {
        *current_output = (block_sum[0] / 2.0f) * scaling_factor;
    }
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    float scaling_factor) {
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    dim3 grid_dim(batch_size);
    dim3 block_dim(256);
    
    size_t shared_mem_size = min(1024, hidden_size) * sizeof(float);
    
    fused_operation_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        input_size,
        hidden_size
    );
    
    return output;
}
"""

fused_operation_cpp_source = """
torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor weight, float scaling_factor);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_operation_cpp_source,
    cuda_sources=fused_operation_source,
    functions=["fused_operation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_operation_cuda(x, self.weight, self.scaling_factor)