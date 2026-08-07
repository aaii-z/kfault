import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear (GEMM) + Bias + ReLU + Division
# We use torch::mm for the matrix multiplication to ensure stability and performance,
# and a custom kernel for the element-wise post-processing (Bias + ReLU + Division).
fused_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_relu_div_kernel(float* __restrict__ out, const float* __restrict__ bias, int M, int N, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;
    
    if (idx < total_elements) {
        int m = idx % M; // Column index corresponds to bias index
        float val = out[idx];
        val += bias[m];
        val = fmaxf(0.0f, val); // ReLU
        out[idx] = val / divisor;
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor) {
    // Perform GEMM: output = input * weight^T
    // torch::mm handles the efficient cuBLAS call internally
    auto output = torch::mm(input, weight.t());
    
    int N = output.size(0); // Batch size
    int M = output.size(1); // Out features
    
    int total_elements = N * M;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    bias_relu_div_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        M, N, divisor
    );
    
    return output;
}
"""

fused_linear_cpp_source = (
    "torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);"
)

# Compile the inline CUDA code
fused_linear_op = load_inline(
    name="fused_linear",
    cpp_sources=fused_linear_cpp_source,
    cuda_sources=fused_linear_source,
    functions=["fused_forward_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for fused Linear + ReLU + Division.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self.fused_op = fused_linear_op

    def forward(self, x):
        return self.fused_op.fused_forward_cuda(x, self.linear.weight, self.linear.bias, self.divisor)