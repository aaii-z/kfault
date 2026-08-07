import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_linear_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float divisor,
    int batch_size,
    int in_features,
    int out_features
) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    float sum = 0.0f;
    
    for (int tile_k = 0; tile_k < (in_features + TILE_SIZE - 1) / TILE_SIZE; ++tile_k) {
        if (i < batch_size && tile_k * TILE_SIZE + threadIdx.y < in_features) {
            shared_A[threadIdx.x][threadIdx.y] = input[i * in_features + tile_k * TILE_SIZE + threadIdx.y];
        } else {
            shared_A[threadIdx.x][threadIdx.y] = 0.0f;
        }
        
        if (j < out_features && tile_k * TILE_SIZE + threadIdx.x < in_features) {
            shared_B[threadIdx.x][threadIdx.y] = weight[j * in_features + tile_k * TILE_SIZE + threadIdx.x];
        } else {
            shared_B[threadIdx.x][threadIdx.y] = 0.0f;
        }
        
        __syncthreads();
        
        if (i < batch_size && j < out_features) {
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += shared_A[threadIdx.x][k] * shared_B[k][threadIdx.y];
            }
        }
        
        __syncthreads();
    }
    
    if (i < batch_size && j < out_features) {
        float val = sum + bias[j];
        val = (val > 0.0f) ? val : 0.0f;
        output[i * out_features + j] = val / divisor;
    }
}

torch::Tensor fused_linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    int batch_size,
    int in_features,
    int out_features
) {
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((batch_size + TILE_SIZE - 1) / TILE_SIZE, 
                  (out_features + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_linear_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    int batch_size,
    int in_features,
    int out_features
);
"""

fused_linear = load_inline(
    name="fused_linear",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divisor = divisor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)
        self.fused_linear = fused_linear

    def forward(self, x):
        return self.fused_linear.fused_linear_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.divisor,
            x.size(0),
            self.in_features,
            self.out_features
        )