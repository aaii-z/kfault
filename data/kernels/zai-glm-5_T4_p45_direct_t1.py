import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Gemm + Sigmoid + Gemm + LogSumExp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for Bias Add + Sigmoid
__global__ void bias_add_sigmoid_kernel(const float* x, const float* bias, float* out, int size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int bias_idx = idx % hidden_size;
        float val = x[idx] + bias[bias_idx];
        out[idx] = 1.0f / (1.0f + expf(-val));
    }
}

// Kernel for Bias Add + LogSumExp
__global__ void bias_add_logsumexp_kernel(const float* x, const float* bias, float* out, int num_rows, int num_cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float max_val = -INFINITY;
    // Each thread finds max over its assigned elements
    for (int i = tid; i < num_cols; i += blockSize) {
        float val = x[row * num_cols + i] + bias[i];
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Reduction to find max
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    
    // Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = tid; i < num_cols; i += blockSize) {
        float val = x[row * num_cols + i] + bias[i];
        sum_exp += expf(val - row_max);
    }
    sdata[tid] = sum_exp;
    __syncthreads();

    // Reduction to sum exp
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out[row] = logf(sdata[0]) + row_max;
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2) {
    auto batch_size = x.size(0);
    auto hidden_size = w1.size(0);
    auto output_size = w2.size(0);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    
    // 1. Linear 1: x @ w1.T
    // We use torch::mm for matrix multiplication (calls cuBLAS)
    auto gemm1_out = torch::mm(x, w1.transpose(0, 1));
    
    // 2. Fused Bias Add + Sigmoid
    auto sigmoid_out = torch::empty({batch_size, hidden_size}, options);
    int size1 = batch_size * hidden_size;
    const int block_size_1 = 256;
    int num_blocks_1 = (size1 + block_size_1 - 1) / block_size_1;
    
    bias_add_sigmoid_kernel<<<num_blocks_1, block_size_1>>>(
        gemm1_out.data_ptr<float>(), 
        b1.data_ptr<float>(), 
        sigmoid_out.data_ptr<float>(), 
        size1, 
        hidden_size
    );
    
    // 3. Linear 2: sigmoid_out @ w2.T
    auto gemm2_out = torch::mm(sigmoid_out, w2.transpose(0, 1));
    
    // 4. Fused Bias Add + LogSumExp
    auto result = torch::empty({batch_size}, options);
    
    int num_rows = batch_size;
    int num_cols = output_size;
    
    int threads_per_block = 256; 
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    bias_add_logsumexp_kernel<<<num_rows, threads_per_block, shared_mem_size>>>(
        gemm2_out.data_ptr<float>(), 
        b2.data_ptr<float>(), 
        result.data_ptr<float>(), 
        num_rows, 
        num_cols
    );
    
    return result;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        # We extract weights and biases to pass to the CUDA kernel
        # Note: nn.Linear stores weights as (out_features, in_features)
        w1 = self.linear1.weight
        b1 = self.linear1.bias
        w2 = self.linear2.weight
        b2 = self.linear2.bias
        
        return self.fused_ops.fused_forward_cuda(x, w1, b1, w2, b2)