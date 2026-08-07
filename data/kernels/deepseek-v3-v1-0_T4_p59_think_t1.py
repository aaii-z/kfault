import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + Swish + scaling
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16

__global__ void linear_swish_scale_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float scaling_factor) {
    
    // Shared memory for tile of input and weight matrices
    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float acc = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load input tile
        int input_col = t * TILE_SIZE + threadIdx.x;
        if (row < batch_size && input_col < in_features) {
            input_tile[threadIdx.y][threadIdx.x] = input[row * in_features + input_col];
        } else {
            input_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load weight tile
        int weight_row = t * TILE_SIZE + threadIdx.y;
        if (col < out_features && weight_row < in_features) {
            weight_tile[threadIdx.y][threadIdx.x] = weight[col * in_features + weight_row];
        } else {
            weight_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += input_tile[threadIdx.y][k] * weight_tile[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Add bias and apply Swish activation and scaling
    if (row < batch_size && col < out_features) {
        acc += bias[col];
        float swish = acc * (1.0f / (1.0f + expf(-acc)));
        output[row * out_features + col] = swish * scaling_factor;
    }
}

torch::Tensor linear_swish_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 blocks(
        (out_features + TILE_SIZE - 1) / TILE_SIZE,
        (batch_size + TILE_SIZE - 1) / TILE_SIZE
    );
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    linear_swish_scale_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor linear_swish_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor);"

# Load the custom CUDA extension
linear_swish_scale = load_inline(
    name="linear_swish_scale",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["linear_swish_scale_cuda"],
    verbose=True
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
        
        # Initialize parameters using same method as nn.Linear
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return linear_swish_scale.linear_swish_scale_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.scaling_factor
        )