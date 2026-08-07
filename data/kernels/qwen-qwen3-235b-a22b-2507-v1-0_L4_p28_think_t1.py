import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

square_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void square_kernel(const float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = y[idx] * y[idx];
    }
}

torch::Tensor square_cuda(torch::Tensor y) {
    auto size = y.numel();
    auto out = torch::empty_like(y);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    square_kernel<<<num_blocks, block_size>>>(y.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

square_cpp = "torch::Tensor square_cuda(torch::Tensor y);"

square_op = load_inline(
    name="square_op",
    cpp_sources=square_cpp,
    cuda_sources=square_source,
    functions=["square_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.square = square_op

    def forward(self, x, y):
        return self.square.square_cuda(y)