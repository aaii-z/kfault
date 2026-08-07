import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused operations
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_linear_bn_swish_kernel(
    const float* input,
    const float* weight,
    const float* bias_linear,
    const float* bn_weight,
    const float* bn_bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    const float* extra_bias,
    float divide_value,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k += TILE_SIZE) {
            if (k + threadIdx.x < in_features) {
                int weight_idx = col * in_features + k + threadIdx.x;
                int input_idx = row * in_features + k + threadIdx.x;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
        __shared__ float shared_sum[BLOCK_SIZE][BLOCK_SIZE];
        shared_sum[threadIdx.y][threadIdx.x] = sum;
        __syncthreads();

        for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                shared_sum[threadIdx.y][threadIdx.x] += shared_sum[threadIdx.y][threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float linear_output = shared_sum[threadIdx.y][0] + bias_linear[col];
            
            // BatchNorm
            float bn_output = (linear_output - running_mean[col]) / sqrtf(running_var[col] + eps);
            bn_output = bn_output * bn_weight[col] + bn_bias[col];
            
            // Bias addition and division
            float biased = bn_output + extra_bias[0];
            float divided = biased / divide_value;
            
            // Swish activation
            output[row * out_features + col] = divided * sigmoid(divided);
        }
    }
}

torch::Tensor fused_linear_bn_swish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_linear,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    torch::Tensor extra_bias,
    float divide_value
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 blocks((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    fused_linear_bn_swish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_linear.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        extra_bias.data_ptr<float>(),
        divide_value,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

cpp_source = "torch::Tensor fused_linear_bn_swish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias_linear, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor running_mean, torch::Tensor running_var, float eps, torch::Tensor extra_bias, float divide_value);"

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_linear_bn_swish_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.eps = bn_eps

    def forward(self, x):
        return fused_ops.fused_linear_bn_swish_cuda(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.eps,
            self.bias,
            self.divide_value
        )