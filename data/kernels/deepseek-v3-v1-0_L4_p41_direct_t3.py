import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused GEMM + BatchNorm + GELU + ReLU CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_gemm_bn_gelu_relu_kernel(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var,
    const float* weight_bn, const float* bias_bn,
    float* output, int batch_size, int in_features, int out_features,
    float eps) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        // GEMM: input[row, :] * weight[:, col] + bias[col]
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[k * out_features + col];
        }
        sum += bias[col];
        
        // BatchNorm: (sum - running_mean[col]) / sqrt(running_var[col] + eps) * weight_bn[col] + bias_bn[col]
        float bn_result = (sum - running_mean[col]) / sqrtf(running_var[col] + eps) * weight_bn[col] + bias_bn[col];
        
        // GELU + ReLU
        float gelu_result = gelu(bn_result);
        float relu_result = fmaxf(0.0f, gelu_result);
        
        output[row * out_features + col] = relu_result;
    }
}

torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight_bn, torch::Tensor bias_bn,
    float eps) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);
    CHECK_INPUT(weight_bn);
    CHECK_INPUT(bias_bn);
    
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size((out_features + block_size.x - 1) / block_size.x,
                   (batch_size + block_size.y - 1) / block_size.y);
    
    fused_gemm_bn_gelu_relu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        weight_bn.data_ptr<float>(), bias_bn.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features,
        eps);
    
    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_gemm_bn_gelu_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight_bn, torch::Tensor bias_bn,
    float eps);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_gemm_bn_gelu_relu",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_bn_gelu_relu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # GEMM parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # BatchNorm parameters
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.weight_bn = nn.Parameter(torch.Tensor(out_features))
        self.bias_bn = nn.Parameter(torch.Tensor(out_features))
        
        # BatchNorm hyperparameters
        self.eps = 1e-5
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.ones_(self.weight_bn)
        nn.init.zeros_(self.bias_bn)
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
    
    def forward(self, x):
        return fused_op.fused_gemm_bn_gelu_relu_cuda(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.weight_bn, self.bias_bn,
            self.eps
        )