import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear + Division + GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 32

__device__ float gelu_approx(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float c = 0.7978845608028654f; // sqrt(2/pi)
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_linear_div_gelu_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, int input_size, int output_size, float divisor) {
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE + 1]; // Padding to avoid bank conflicts
    
    float val = 0.0f;
    
    for (int k = 0; k < input_size; k += BLOCK_SIZE) {
        // Load tile from A (input)
        if (row < batch_size && (k + tx) < input_size) {
            shared_A[ty][tx] = input[row * input_size + k + tx];
        } else {
            shared_A[ty][tx] = 0.0f;
        }
        
        // Load tile from B (weight)
        // Weight is (output_size, input_size)
        // We need weight[col][k + ty]
        if (col < output_size && (k + ty) < input_size) {
            shared_B[tx][ty] = weight[col * input_size + k + ty];
        } else {
            shared_B[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            val += shared_A[ty][i] * shared_B[tx][i];
        }
        
        __syncthreads();
    }
    
    if (row < batch_size && col < output_size) {
        float b = bias[col];
        val += b;
        val /= divisor;
        val = gelu_approx(val);
        output[row * output_size + col] = val;
    }
}

torch::Tensor fused_linear_div_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, input.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((output_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    fused_linear_div_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, input_size, output_size, divisor
    );
    
    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_linear_div_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_div_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_linear_div_gelu_cuda(x, self.linear.weight, self.linear.bias, self.divisor)