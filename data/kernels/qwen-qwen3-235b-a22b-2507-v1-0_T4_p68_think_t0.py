import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_min_sub_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_min_sub_kernel(const float* s, float* out, float c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = s[idx] - c;
        out[idx] = (val < 0.0f) ? val : 0.0f;
    }
}

torch::Tensor fused_min_sub_cuda(torch::Tensor s, torch::Tensor c) {
    float c_val = c.item<float>();
    auto size = s.numel();
    auto out = torch::empty_like(s);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_min_sub_kernel<<<num_blocks, block_size>>>(s.data_ptr<float>(), out.data_ptr<float>(), c_val, size);

    return out;
}
"""

fused_min_sub_cpp = "torch::Tensor fused_min_sub_cuda(torch::Tensor s, torch::Tensor c);"

fused_min_sub = load_inline(
    name="fused_min_sub",
    cpp_sources=fused_min_sub_cpp,
    cuda_sources=fused_min_sub_source,
    functions=["fused_min_sub_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses min and subtraction operations into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.fused_min_sub = fused_min_sub

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_min_sub.fused_min_sub_cuda(x, self.constant)
        return x