import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + relu + division
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_matmul_relu_div_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int M, int N, int K, float divisor) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += input[row * K + k] * weight[k * N + col];
        }
        
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // Apply ReLU and division
        sum = fmaxf(0.0f, sum) / divisor;
        output[row * N + col] = sum;
    }
}

torch::Tensor fused_matmul_relu_div_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor) {
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(1);
    
    auto output = torch::empty({M, N}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (M + block_size.y - 1) / block_size.y);
    
    fused_matmul_relu_div_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        M, N, K, divisor);
    
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_matmul_relu_div_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_matmul_relu_div_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.divisor = divisor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.fused_op = fused_operator
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_op.fused_matmul_relu_div_cuda(x, self.weight, self.bias, self.divisor)