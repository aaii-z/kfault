import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Gemm + LogSumExp + LeakyReLU + GELU
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

constexpr float negative_slope = 0.01f;
constexpr float sqrt_2_over_pi = 0.7978845608028654f;
constexpr float gelu_coeff = 0.044715f;

__device__ float leaky_relu(float x) {
    return x > 0.0f ? x : negative_slope * x;
}

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + gelu_coeff * x * x)));
}

__global__ void fused_gemm_activation_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[k * out_features + col];
        }
        if (bias) {
            sum += bias[col];
        }
        
        // Store intermediate result for LogSumExp
        __shared__ float row_max[256];
        __shared__ float row_sum[256];
        
        // First: LogSumExp reduction within thread block
        float max_val = -FLT_MAX;
        float exp_sum = 0.0f;
        
        for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
            float val = output[row * out_features + i];
            max_val = fmaxf(max_val, val);
        }
        
        __shared__ float shared_max[256];
        if (threadIdx.x == 0) {
            shared_max[threadIdx.y] = -FLT_MAX;
        }
        __syncthreads();
        
        atomicMax((int*)&shared_max[threadIdx.y], __float_as_int(max_val));
        __syncthreads();
        max_val = __int_as_float(*(int*)&shared_max[threadIdx.y]);
        
        for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
            float val = output[row * out_features + i];
            exp_sum += expf(val - max_val);
        }
        
        __shared__ float shared_sum[256];
        if (threadIdx.x == 0) {
            shared_sum[threadIdx.y] = 0.0f;
        }
        __syncthreads();
        
        atomicAdd(&shared_sum[threadIdx.y], exp_sum);
        __syncthreads();
        exp_sum = shared_sum[threadIdx.y];
        
        float logsumexp = max_val + logf(exp_sum);
        
        // Apply activations
        float result = logsumexp;
        result = leaky_relu(result);
        result = leaky_relu(result);
        result = gelu(result);
        result = gelu(result);
        
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            output[row] = result;
        }
    }
}

torch::Tensor fused_operator_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(1);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    
    fused_gemm_activation_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_operator_cpp_source = "torch::Tensor fused_operator_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.fused_op = fused_operator

    def forward(self, x):
        return self.fused_op.fused_operator_cuda(x, self.weight, self.bias if self.bias is not None else torch.Tensor())