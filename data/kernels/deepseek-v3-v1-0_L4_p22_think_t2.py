import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__ void fused_linear_scale_clamp_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size,
    float scale_factor,
    float clamp_min,
    float clamp_max) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[row * input_size + i] * weight[col * input_size + i];
        }
        if (bias) {
            sum += bias[col];
        }
        
        sum = sum * scale_factor * 2.0f;
        
        if (sum < clamp_min) sum = clamp_min;
        if (sum > clamp_max) sum = clamp_max;
        
        output[row * hidden_size + col] = sum;
    }
}

torch::Tensor fused_linear_scale_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);
    
    auto output = torch::empty({batch_size, hidden_size}, input.options());
    
    dim3 blocks((hidden_size + TILE_DIM - 1) / TILE_DIM,
                (batch_size + TILE_DIM - 1) / TILE_DIM);
    dim3 threads(TILE_DIM, TILE_DIM);
    
    fused_linear_scale_clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        scale_factor,
        clamp_min,
        clamp_max);
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_linear_scale_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max);
"""

# Load the custom CUDA operations
custom_ops = load_inline(
    name='custom_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_linear_scale_clamp_cuda'],
    verbose=True,
    extra_cflags=['-O2'],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = custom_ops.fused_linear_scale_clamp_cuda(
            x, self.weight, self.bias,
            self.scale_factor, self.clamp_min, self.clamp_max
        )
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.nn.functional.mish(x)
        return x