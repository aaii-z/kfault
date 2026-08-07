import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for matmul + bn + bias + divide + swish
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_operator_kernel(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var,
    const float* bn_weight, const float* bn_bias,
    const float* add_bias, float divide_value,
    float eps, 
    float* output, int batch_size, int in_features, int out_features) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        // Matmul: x * weight^T
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        // Add linear bias
        sum += bias[col];
        
        // Batch normalization
        float mean = running_mean[col];
        float var = running_var[col];
        float normalized = (sum - mean) / sqrtf(var + eps);
        float bn_result = normalized * bn_weight[col] + bn_bias[col];
        
        // Add custom bias
        bn_result += add_bias[0];
        
        // Divide
        bn_result /= divide_value;
        
        // Swish activation: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + expf(-bn_result));
        output[row * out_features + col] = bn_result * sigmoid_val;
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor add_bias, float divide_value,
    float eps) {
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size((out_features + block_size.x - 1) / block_size.x,
                  (batch_size + block_size.y - 1) / block_size.y);
    
    fused_operator_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        add_bias.data_ptr<float>(), divide_value,
        eps, output.data_ptr<float>(),
        batch_size, in_features, out_features);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor add_bias, float divide_value,
    float eps);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.divide_value = divide_value
        
        # Linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
        # Custom bias
        self.add_bias = nn.Parameter(torch.randn(bias_shape))
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.bn_weight, self.bn_bias,
            self.add_bias, self.divide_value,
            self.bn_eps
        )