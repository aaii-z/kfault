import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused kernel: BatchNorm + Bias + Divide + Swish
// Assumes input x is the output of a Linear layer (N, C)
// BatchNorm parameters (gamma, beta, mean, var) are vectors of size C
// Bias is broadcasted (size 1 or C)
// Divide value is a scalar
__global__ void fused_bn_bias_div_swish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int num_rows, int num_cols, float eps, float divide_val) {

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        int idx = row * num_cols + col;
        
        // BatchNorm
        float val = x[idx];
        float mean = running_mean[col];
        float var = running_var[col];
        float scale = gamma[col];
        float shift = beta[col];
        
        // (x - mean) / sqrt(var + eps) * gamma + beta
        float norm = (val - mean) * rsqrtf(var + eps) * scale + shift;
        
        // Add Bias
        norm += bias[0]; // Assuming bias shape is (1,)
        
        // Divide
        norm /= divide_val;
        
        // Swish: x * sigmoid(x)
        float sig = 1.0f / (1.0f + expf(-norm));
        out[idx] = norm * sig;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor bias,
    float eps,
    float divide_val) {

    auto sizes = x.sizes();
    int num_rows = sizes[0];
    int num_cols = sizes[1];
    
    auto out = torch::empty_like(x);

    // Grid configuration
    // We want to cover N x C elements.
    // We launch a 2D grid. 
    // Y dimension handles rows (N).
    // X dimension handles columns (C).
    
    const int threads_per_block = 256;
    int blocks_x = (num_cols + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_x, num_rows);
    dim3 block(threads_per_block);

    fused_bn_bias_div_swish_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        num_rows, num_cols, eps, divide_val
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor bias, float eps, float divide_val);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication followed by a fused
    BatchNorm, Bias, Division, and Swish activation using a custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1. Matrix Multiplication (using PyTorch's optimized Linear)
        x = self.matmul(x)
        
        # 2. Fused Operations (BatchNorm + Bias + Div + Swish)
        # We extract the parameters from the BatchNorm layer
        # Note: BatchNorm1d stores running_mean, running_var, weight (gamma), bias (beta)
        # We need to ensure the tensors are contiguous for the CUDA kernel
        gamma = self.bn.weight
        beta = self.bn.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        
        x = self.fused_ops.fused_ops_cuda(
            x, gamma, beta, running_mean, running_var, self.bias, self.bn.eps, self.divide_value
        )
        
        return x