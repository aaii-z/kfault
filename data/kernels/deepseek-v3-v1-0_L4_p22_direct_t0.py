import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for the entire forward pass
fused_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ float mish_activation(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_forward_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float scale_factor,
    float clamp_min,
    float clamp_max,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < hidden_size) {
        // Matrix multiplication
        float sum = 0.0f;
        for (int k = 0; k < input_size; ++k) {
            sum += input[row * input_size + k] * weight[col * input_size + k];
        }
        if (bias) {
            sum += bias[col];
        }
        
        // Scale
        sum *= scale_factor;
        
        // Residual connection (x + x)
        sum += sum;
        
        // Clamp
        sum = fminf(fmaxf(sum, clamp_min), clamp_max);
        
        // Store intermediate result for LogSumExp
        output[row * hidden_size + col] = sum;
    }
}

__global__ void logsumexp_and_mish_kernel(
    const float* intermediate,
    float* output,
    int batch_size,
    int hidden_size
) {
    int row = blockIdx.x;
    
    if (row < batch_size) {
        // LogSumExp
        float max_val = -INFINITY;
        for (int col = 0; col < hidden_size; ++col) {
            max_val = fmaxf(max_val, intermediate[row * hidden_size + col]);
        }
        
        float sum_exp = 0.0f;
        for (int col = 0; col < hidden_size; ++col) {
            sum_exp += expf(intermediate[row * hidden_size + col] - max_val);
        }
        
        float logsumexp = logf(sum_exp) + max_val;
        
        // Mish activation and final multiplication
        float mish_val = mish_activation(logsumexp);
        output[row] = logsumexp * mish_val;
    }
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max
) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);
    
    auto intermediate = torch::empty({batch_size, hidden_size}, input.options());
    auto output = torch::empty({batch_size}, input.options());
    
    // Launch matrix multiplication + scaling + residual + clamp kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim((hidden_size + block_dim.x - 1) / block_dim.x, 
                  (batch_size + block_dim.y - 1) / block_dim.y);
    
    fused_forward_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        scale_factor,
        clamp_min,
        clamp_max,
        intermediate.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );
    
    // Launch LogSumExp + Mish kernel
    logsumexp_and_mish_kernel<<<batch_size, 1>>>(
        intermediate.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size
    );
    
    return output.unsqueeze(1);  // Add keepdim dimension
}
"""

fused_forward_cpp_source = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max
);
"""

# Compile the inline CUDA code
fused_forward = load_inline(
    name="fused_forward",
    cpp_sources=fused_forward_cpp_source,
    cuda_sources=fused_forward_source,
    functions=["fused_forward_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_forward = fused_forward
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_forward.fused_forward_cuda(
            x, self.weight, self.bias, self.scale_factor, self.clamp_min, self.clamp_max
        )