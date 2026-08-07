import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused linear + division + GELU
fused_linear_div_gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32

__device__ float gelu_approximation(float x) {
    return 0.5 * x * (1.0 + tanhf(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

__global__ void fused_linear_div_gelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const float divisor,
    int M, int K, int N
) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];

    float sum = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int input_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && input_col < K) {
            input_tile[threadIdx.y][threadIdx.x] = input[row * K + input_col];
        } else {
            input_tile[threadIdx.y][threadIdx.x] = 0.0;
        }

        int weight_row = t * TILE_SIZE + threadIdx.y;
        if (weight_row < K && col < N) {
            weight_tile[threadIdx.y][threadIdx.x] = weight[weight_row * N + col];
        } else {
            weight_tile[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += input_tile[threadIdx.y][i] * weight_tile[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        sum += bias[col];
        sum /= divisor;
        sum = gelu_approximation(sum);
        output[row * N + col] = sum;
    }
}

torch::Tensor fused_linear_div_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(1);

    auto output = torch::zeros({M, N}, input.options());

    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    fused_linear_div_gelu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        M, K, N
    );

    return output;
}
"""

fused_linear_div_gelu_cpp_source = """
torch::Tensor fused_linear_div_gelu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);
"""

fused_linear_div_gelu = load_inline(
    name='fused_linear_div_gelu',
    cpp_sources=fused_linear_div_gelu_cpp_source,
    cuda_sources=fused_linear_div_gelu_cuda_source,
    functions=['fused_linear_div_gelu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.divisor = divisor

        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.register_buffer('weight_transposed', torch.Tensor(input_size, output_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.weight_transposed.copy_(self.weight.t())

    def forward(self, x):
        return fused_linear_div_gelu.fused_linear_div_gelu_cuda(x, self.weight_transposed, self.bias, self.divisor)