import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_scale_kernel(const float* y, float* out, float scaling_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = y[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        out[idx] = val * sigmoid_val * scaling_factor;
    }
}

torch::Tensor swish_scale_cuda(torch::Tensor y, float scaling_factor) {
    auto size = y.numel();
    auto out = torch::empty_like(y);

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    swish_scale_kernel<<<num_blocks, block_size>>>(y.data_ptr<float>(), out.data_ptr<float>(), scaling_factor, size);

    return out;
}
"""

swish_scale_cpp = "torch::Tensor swish_scale_cuda(torch::Tensor y, float scaling_factor);"

swish_scale = load_inline(
    name="swish_scale",
    cpp_sources=swish_scale_cpp,
    cuda_sources=swish_scale_source,
    functions=["swish_scale_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.swish_scale = swish_scale

    def forward(self, x):
        x = self.matmul(x)
        x = self.swish_scale.swish_scale_cuda(x, self.scaling_factor)
        return x