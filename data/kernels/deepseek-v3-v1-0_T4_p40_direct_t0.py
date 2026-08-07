import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + scaling + residual addition
fused_operation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_matmul_scale_add_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float scaling_factor, int batch_size, 
    int in_features, int out_features) {
    
    extern __shared__ float shared_mem[];
    float* weight_shared = shared_mem;
    float* input_shared = shared_mem + out_features * blockDim.x;
    
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    // Load weight matrix into shared memory (coalesced access)
    if (col < out_features) {
        for (int k = 0; k < (in_features + blockDim.x - 1) / blockDim.x; k++) {
            int weight_idx = k * blockDim.x + col;
            if (weight_idx < in_features) {
                weight_shared[col * blockDim.x + k] = weight[col * in_features + weight_idx];
            }
        }
    }
    __syncthreads();
    
    // Perform matrix multiplication
    float result = 0.0f;
    for (int k = 0; k < in_features; k += blockDim.x) {
        int input_idx = k + col;
        if (input_idx < in_features) {
            input_shared[col] = input[row * in_features + input_idx];
        }
        __syncthreads();
        
        for (int j = 0; j < min(blockDim.x, in_features - k); j++) {
            result += input_shared[j] * weight_shared[col * blockDim.x + j];
        }
        __syncthreads();
    }
    
    // Add bias and apply scaling
    if (col < out_features) {
        float biased_result = result + (bias ? bias[col] : 0.0f);
        output[row * out_features + col] = biased_result * scaling_factor + biased_result;
    }
}

torch::Tensor fused_matmul_scale_add_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    float scaling_factor) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_dim(256);
    dim3 grid_dim(batch_size);
    
    size_t shared_mem_size = (out_features * block_dim.x + block_dim.x) * sizeof(float);
    
    fused_matmul_scale_add_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_operation_cpp_source = """
torch::Tensor fused_matmul_scale_add_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    float scaling_factor);
"""

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_operation_cpp_source,
    cuda_sources=fused_operation_source,
    functions=["fused_matmul_scale_add_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        
        self.fused_op = fused_operation

    def forward(self, x):
        return self.fused_op.fused_matmul_scale_add_cuda(
            x, self.weight, self.bias, self.scaling_factor
        )