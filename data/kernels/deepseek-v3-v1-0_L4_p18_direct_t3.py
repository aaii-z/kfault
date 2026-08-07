import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_ops_kernel(const float* input, const float* weight, const float* bias, 
                                float* output, int batch_size, int in_features, int out_features) {
    extern __shared__ float shared_mem[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < batch_size) {
        // Each block processes one row of the input
        float* row_sum = shared_mem;
        
        // Initialize row sum
        if (tid == 0) {
            row_sum[0] = 0.0f;
        }
        __syncthreads();
        
        // Parallel matrix multiplication and reduction
        for (int col_start = 0; col_start < out_features; col_start += blockDim.x) {
            int col = col_start + tid;
            float dot_product = 0.0f;
            
            if (col < out_features) {
                // Matrix multiplication for this column
                for (int k = 0; k < in_features; k++) {
                    dot_product += input[row * in_features + k] * weight[k * out_features + col];
                }
                dot_product += bias[col];
            }
            
            // Atomic add to row sum
            if (col < out_features) {
                atomicAdd(row_sum, dot_product);
            }
        }
        __syncthreads();
        
        // Store the final result
        if (tid == 0) {
            output[row] = row_sum[0];
        }
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(1);
    
    auto output = torch::zeros({batch_size, 1}, torch::dtype(torch::kFloat32).device(input.device()));
    
    int block_size = 1024;
    int grid_size = batch_size;
    int shared_mem_size = sizeof(float);
    
    fused_ops_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(x, self.weight, self.bias)