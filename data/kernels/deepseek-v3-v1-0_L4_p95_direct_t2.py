import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + add + swish + tanh + gelu + hardtanh
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr float GELU_COEF_A = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

__device__ __forceinline__ float swish_activation(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

__device__ __forceinline__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__global__ void fused_activation_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    const float* add_value,
    float* output, 
    int batch_size, 
    int in_features, 
    int out_features,
    float min_val,
    float max_val) {
    
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
        
        sum += add_value[col];
        
        // Swish: sigmoid(x) * x
        float swish_result = swish_activation(sum);
        
        // Tanh
        float tanh_result = tanhf(swish_result);
        
        // GELU
        float gelu_result = gelu_activation(tanh_result);
        
        // Hardtanh
        float final_result = fminf(fmaxf(gelu_result, min_val), max_val);
        
        output[row * out_features + col] = final_result;
    }
}

torch::Tensor fused_activation_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor add_value,
    float min_val,
    float max_val) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(1);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size((out_features + block_size.x - 1) / block_size.x, 
                   (batch_size + block_size.y - 1) / block_size.y);
    
    fused_activation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        add_value.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        min_val,
        max_val
    );
    
    return output;
}
"""

fused_activation_cpp_source = """
torch::Tensor fused_activation_forward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor add_value,
    float min_val,
    float max_val);
"""

# Compile the inline CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store weight and bias as parameters for the linear layer
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        
        self.min_val = -1.0
        self.max_val = 1.0
        self.fused_activation = fused_activation

    def forward(self, x):
        return self.fused_activation.fused_activation_forward(
            x, self.weight, self.bias, self.add_value, self.min_val, self.max_val
        )