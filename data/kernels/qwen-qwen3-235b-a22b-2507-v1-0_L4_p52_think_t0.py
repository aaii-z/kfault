import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_activation_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_activation_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float sp;
        if (val > 20.0f) {
            sp = val;
        } else {
            sp = log1pf(expf(val));
        }
        float tanh_sp = tanhf(sp);
        out[idx] = tanh_sp * val;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_activation_kernel<<<num_blocks, block_size, 0, stream>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
"""

fused_activation_cpp = "torch::Tensor fused_activation_cuda(torch::Tensor x);"

fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused activation kernel (softplus + tanh + multiply)
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_activation_cuda(x)
        x = self.bn(x)
        return x