import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__global__ void fused_matmul_avg_gelu_scale_max_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int pool_kernel_size,
    float scale_factor
) {
    extern __shared__ float shared_mem[];
    float* shared_vals = shared_mem;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_windows = out_features / pool_kernel_size;
    
    // Each block processes one batch element
    if (bid >= batch_size) return;
    
    // Initialize shared memory for max reduction
    if (tid < num_windows) {
        shared_vals[tid] = -FLT_MAX;
    }
    __syncthreads();
    
    // Compute linear layer and average pool for each window
    for (int window = tid; window < num_windows; window += blockDim.x) {
        float sum = 0.0f;
        int start_idx = window * pool_kernel_size;
        
        // Compute matmul and average pool
        for (int j = 0; j < pool_kernel_size; ++j) {
            int weight_idx = start_idx + j;
            float val = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                val += input[bid * in_features + k] * weight[weight_idx * in_features + k];
            }
            if (bias) val += bias[weight_idx];
            sum += val;
        }
        
        // Apply average pooling, GELU, and scaling
        float avg_val = sum / pool_kernel_size;
        float activated = gelu(avg_val);
        float scaled = activated * scale_factor;
        
        // Store result for max reduction
        atomicMax(&shared_vals[window], scaled);
    }
    __syncthreads();
    
    // Max reduction across windows
    if (tid == 0) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < num_windows; ++i) {
            if (shared_vals[i] > max_val) {
                max_val = shared_vals[i];
            }
        }
        output[bid] = max_val;
    }
}

torch::Tensor fused_operation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int pool_kernel_size,
    float scale_factor
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    auto output = torch::empty({batch_size}, input.options());
    
    int num_windows = out_features / pool_kernel_size;
    int shared_mem_size = num_windows * sizeof(float);
    
    dim3 blocks(batch_size);
    dim3 threads(256);
    
    fused_matmul_avg_gelu_scale_max_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        pool_kernel_size,
        scale_factor
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_operation_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int pool_kernel_size,
    float scale_factor
);
"""

# Load the custom CUDA operation
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=['fused_operation_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        # Initialize linear layer parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_ops.fused_operation_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.pool_kernel_size, 
            self.scale_factor
        )