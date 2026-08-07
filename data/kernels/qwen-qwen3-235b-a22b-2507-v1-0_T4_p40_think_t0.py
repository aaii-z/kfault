import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_scale_kernel(const float* x, float* out, float factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] * factor;
    }
}

torch::Tensor elementwise_scale_cuda(torch::Tensor x, float factor) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_scale_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), factor, size);

    return out;
}
"""

elementwise_scale_cpp_source = (
    "torch::Tensor elementwise_scale_cuda(torch::Tensor x, float factor);"
)

elementwise_scale = load_inline(
    name="elementwise_scale",
    cpp_sources=elementwise_scale_cpp_source,
    cuda_sources=elementwise_scale_source,
    functions=["elementwise_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.elementwise_scale = elementwise_scale

    def forward(self, x):
        x = self.matmul(x)
        factor = self.scaling_factor + 1.0
        x = self.elementwise_scale.elementwise_scale_cuda(x, factor)
        return x