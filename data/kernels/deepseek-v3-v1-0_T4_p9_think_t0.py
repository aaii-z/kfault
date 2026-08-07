import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sub_mul_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float subtract_val,
    float multiply_val,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        if (bias) {
            sum += bias[col];
        }
        
        sum = (sum - subtract_val) * multiply_val;
        output[row * out_features + col] = sum > 0 ? sum : 0;
    }
}

torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_val,
    float multiply_val
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (out_features + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );

    fused_linear_sub_mul_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract_val,
        multiply_val,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_val,
    float multiply_val
);
"""

# Load the custom CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=['fused_linear_sub_mul_relu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_ops.fused_linear_sub_mul_relu_cuda(
            x,
            self.weight,
            self.bias,
            self.subtract_value,
            self.multiply_value
        )