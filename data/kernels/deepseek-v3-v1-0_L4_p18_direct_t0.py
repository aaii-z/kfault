import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_ops_kernel(const float* input, const float* weight, const float* bias, 
                                float* output, int batch_size, int in_features, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        // Matrix multiplication
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[k * out_features + col];
        }
        sum += bias[col];
        
        // Store result in shared memory for reduction operations
        __shared__ float shared_mem[1024];
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        shared_mem[tid] = sum;
        __syncthreads();
        
        // Sum reduction
        for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_mem[tid] += shared_mem[tid + s];
            }
            __syncthreads();
        }
        
        // Max reduction
        float max_val = shared_mem[0];
        for (int s = 1; s < blockDim.x * blockDim.y; s++) {
            if (shared_mem[s] > max_val) {
                max_val = shared_mem[s];
            }
        }
        
        // Mean calculation
        float mean_val = shared_mem[0] / (blockDim.x * blockDim.y);
        
        // LogSumExp
        float max_exp = max_val;
        float exp_sum = 0.0f;
        for (int s = 0; s < blockDim.x * blockDim.y; s++) {
            exp_sum += expf(shared_mem[s] - max_exp);
        }
        float logsumexp1 = max_exp + logf(exp_sum);
        
        // Second LogSumExp (applied to the result of first)
        float logsumexp2 = logsumexp1;  // For single value, logsumexp is the value itself
        
        if (tid == 0) {
            output[row] = logsumexp2;
        }
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(1);
    
    auto output = torch::zeros({batch_size, 1}, torch::dtype(torch::kFloat32).device(input.device()));
    
    dim3 block_size(32, 32);
    dim3 grid_size((out_features + block_size.x - 1) / block_size.x, 
                   (batch_size + block_size.y - 1) / block_size.y);
    
    fused_ops_kernel<<<grid_size, block_size>>>(
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