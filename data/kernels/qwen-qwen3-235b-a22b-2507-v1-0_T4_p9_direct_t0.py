import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear, subtract, multiply, and ReLU
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sub_mul_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float subtract_value,
    float multiply_value,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    if (idx >= total_elements) return;

    int row = idx / out_features;
    int col = idx % out_features;

    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];

    sum = (sum - subtract_value) * multiply_value;
    sum = fmaxf(0.0f, sum);  // ReLU

    output[idx] = sum;
}

torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_linear_sub_mul_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        subtract_value,
        multiply_value,
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_linear_sub_mul_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_linear_sub_mul_relu",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_linear_sub_mul_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused linear, subtract, multiply, and ReLU in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_linear_sub_mul_relu_cuda(
            x, self.linear.weight, self.linear.bias,
            self.subtract_value, self.multiply_value
        )