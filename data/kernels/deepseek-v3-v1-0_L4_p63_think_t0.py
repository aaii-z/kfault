import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + ReLU + division
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void linear_relu_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int in_features,
    int out_features,
    float divisor,
    int batch_size
) {
    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < in_features; t += TILE_SIZE) {
        // Load input tile
        if (row < batch_size && (t + threadIdx.x) < in_features) {
            input_tile[threadIdx.y][threadIdx.x] = input[row * in_features + t + threadIdx.x];
        } else {
            input_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load weight tile
        if (col < out_features && (t + threadIdx.y) < in_features) {
            weight_tile[threadIdx.y][threadIdx.x] = weight[col * in_features + t + threadIdx.y];
        } else {
            weight_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += input_tile[threadIdx.y][k] * weight_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < batch_size && col < out_features) {
        sum += bias[col];
        sum = fmaxf(0.0f, sum) / divisor;
        output[row * out_features + col] = sum;
    }
}

torch::Tensor linear_relu_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 blocks(
        (out_features + TILE_SIZE - 1) / TILE_SIZE,
        (batch_size + TILE_SIZE - 1) / TILE_SIZE
    );
    dim3 threads(TILE_SIZE, TILE_SIZE);

    linear_relu_div_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        in_features,
        out_features,
        divisor,
        batch_size
    );

    return output;
}
"""

cpp_source = "torch::Tensor linear_relu_div_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);"

# Load the custom CUDA extension
linear_relu_div = load_inline(
    name="linear_relu_div",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["linear_relu_div_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divisor = divisor
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters using same method as nn.Linear
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return linear_relu_div.linear_relu_div_cuda(x, self.weight, self.bias, self.divisor)

import math
batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, divisor]