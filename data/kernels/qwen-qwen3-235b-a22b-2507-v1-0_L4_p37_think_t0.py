import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_bias_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void swish_bias_add_kernel(const float* y, const float* bias, float* out, int total_elements, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int j = idx % out_features;
        float y_val = y[idx];
        float sigmoid_y = 1.0f / (1.0f + expf(-y_val));
        out[idx] = sigmoid_y * y_val + bias[j];
    }
}

torch::Tensor swish_bias_add_cuda(torch::Tensor y, torch::Tensor bias) {
    auto total_elements = y.numel();
    auto out = torch::empty_like(y);
    int out_features = y.size(1);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    swish_bias_add_kernel<<<num_blocks, block_size>>>(y.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), total_elements, out_features);

    return out;
}
"""

swish_bias_add_cpp = "torch::Tensor swish_bias_add_cuda(torch::Tensor y, torch::Tensor bias);"

swish_bias_add = load_inline(
    name="swish_bias_add",
    cpp_sources=swish_bias_add_cpp,
    cuda_sources=swish_bias_add_source,
    functions=["swish_bias_add_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.swish_bias_add_op = swish_bias_add

    def forward(self, x):
        x = self.matmul(x)
        x = self.swish_bias_add_op.swish_bias_add_cuda(x, self.bias)
        x = self.group_norm(x)
        return x