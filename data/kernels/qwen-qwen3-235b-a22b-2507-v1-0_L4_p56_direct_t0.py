import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused linear + sigmoid + sum operation
fused_linear_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sigmoid_sum_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    float sum_val = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float acc = bias[i];
        const float* input_row = input + batch_idx * input_size;
        const float* weight_row = weight + i * input_size;
        
        for (int j = 0; j < input_size; j++) {
            acc += input_row[j] * weight_row[j];
        }

        // Sigmoid activation
        float sigmoid_val = 1.0f / (1.0f + expf(-acc));
        sum_val += sigmoid_val;
    }
    output[batch_idx] = sum_val;
}

torch::Tensor fused_linear_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);

    auto output = torch::empty({batch_size, 1}, input.options());

    const int block_size = 1;
    const int num_blocks = batch_size;

    fused_linear_sigmoid_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return output;
}
"""

fused_linear_sigmoid_sum_cpp_source = """
torch::Tensor fused_linear_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_linear_sigmoid_sum",
    cpp_sources=fused_linear_sigmoid_sum_cpp_source,
    cuda_sources=fused_linear_sigmoid_sum_source,
    functions=["fused_linear_sigmoid_sum_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused linear + sigmoid + sum using custom CUDA kernel.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fused_op = fused_op.fused_linear_sigmoid_sum_cuda

    def forward(self, x):
        return self.fused_op(x, self.linear.weight, self.linear.bias)