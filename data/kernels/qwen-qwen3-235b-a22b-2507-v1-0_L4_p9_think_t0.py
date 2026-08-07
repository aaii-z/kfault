import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_kernel(
    const float* x, 
    const float* weight, 
    const float* bias,
    float subtract_value,
    float multiply_value,
    float* out,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    if (idx >= total_elements) return;

    int i = idx / out_features;
    int j = idx % out_features;
    
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += x[i * in_features + k] * weight[j * in_features + k];
    }
    sum += bias[j];
    sum = (sum - subtract_value) * multiply_value;
    out[idx] = sum > 0 ? sum : 0.0f;
}

torch::Tensor fused_linear_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value
) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = bias.size(0);
    auto out = torch::empty({batch_size, out_features}, x.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;
    
    fused_linear_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract_value,
        multiply_value,
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return out;
}
"""

fused_linear_cpp = "torch::Tensor fused_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float subtract_value, float multiply_value);"

fused_linear = load_inline(
    name="fused_linear",
    cpp_sources=fused_linear_cpp,
    cuda_sources=fused_linear_source,
    functions=["fused_linear_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_linear = fused_linear

    def forward(self, x):
        return self.fused_linear.fused_linear_cuda(x, self.weight, self.bias, self.subtract_value, self.multiply_value)