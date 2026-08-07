import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_linear_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_swish_scale_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ out,
    int batch_size, 
    int in_features, 
    int out_features, 
    float scaling_factor) {
    
    int idx = blockIdx.x;
    if (idx >= batch_size * out_features) return;

    int i = idx / out_features;
    int j = idx % out_features;
    
    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += x[i * in_features + k] * weight[j * in_features + k];
    }
    sum += bias[j];
    
    float sigmoid_val = 1.0f / (1.0f + expf(-sum));
    out[idx] = sum * sigmoid_val * scaling_factor;
}

torch::Tensor fused_linear_swish_scale_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor) {
    
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);

    auto out = torch::empty({batch_size, out_features}, x.options());

    int total_elements = batch_size * out_features;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_linear_swish_scale_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, in_features, out_features, scaling_factor
    );

    return out;
}
"""

fused_linear_swish_scale_cpp = "torch::Tensor fused_linear_swish_scale_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float scaling_factor);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_linear_swish_scale_cpp,
    cuda_sources=fused_linear_swish_scale_source,
    functions=["fused_linear_swish_scale_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--use_fast_math", "-arch=sm_70"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_op.fused_linear_swish_scale_cuda(x, self.weight, self.bias, self.scaling_factor)