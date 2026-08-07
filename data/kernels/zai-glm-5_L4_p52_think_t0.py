import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Mish activation
# Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void mish_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float sp;
        // Compute softplus: ln(1 + exp(x))
        // Use threshold to avoid exp overflow and for efficiency
        if (val > 20.0f) {
            sp = val;
        } else {
            sp = log1pf(expf(val));
        }
        // Compute mish: x * tanh(softplus)
        out[idx] = val * tanhf(sp);
    }
}

torch::Tensor mish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
"""

mish_cpp_source = "torch::Tensor mish_cuda(torch::Tensor x);"

# Compile the inline CUDA code
mish_op = load_inline(
    name="mish_op",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA Mish operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.mish = mish_op

    def forward(self, x):
        x = self.conv(x)
        x = self.mish.mish_cuda(x)
        x = self.bn(x)
        return x