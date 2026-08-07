import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_linear_min_sub_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float constant_val,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch_size * out_features) {
        int row = index / out_features;
        int col = index % out_features;
        
        float value = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            value += input[row * in_features + k] * weight[col * in_features + k];
        }
        value += bias[col];
        value = fminf(value, constant_val);
        value -= constant_val;
        output[index] = value;
    }
}

torch::Tensor fused_linear_min_sub_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float constant_val
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    int total_elements = batch_size * out_features;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_linear_min_sub_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        constant_val,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_linear_min_sub_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float constant_val);
"""

fused_linear_op = load_inline(
    name="fused_linear_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_min_sub_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.constant = nn.Parameter(torch.tensor(constant), requires_grad=False)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        constant_val = self.constant.item()
        return fused_linear_op.fused_linear_min_sub_cuda(x, self.weight, self.bias, constant_val)