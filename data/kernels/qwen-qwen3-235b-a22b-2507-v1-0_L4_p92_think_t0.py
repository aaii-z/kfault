import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_tanh_hardswish_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_tanh_hardswish_add_kernel(
    const float* x_conv,
    const float* x_norm,
    float* x_res,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float norm_val = x_norm[idx];
        float z = tanhf(norm_val);
        float z_plus_3 = z + 3.0f;
        float relu6 = (z_plus_3 < 0.0f) ? 0.0f : ((z_plus_3 > 6.0f) ? 6.0f : z_plus_3);
        float w = z * relu6 / 6.0f;
        x_res[idx] = x_conv[idx] + w;
    }
}

torch::Tensor fused_tanh_hardswish_add_cuda(
    torch::Tensor x_conv,
    torch::Tensor x_norm) {
    
    TORCH_CHECK(x_conv.is_contiguous(), "x_conv must be contiguous");
    TORCH_CHECK(x_norm.is_contiguous(), "x_norm must be contiguous");
    TORCH_CHECK(x_conv.sizes() == x_norm.sizes(), "Input tensors must have same shape");
    
    auto size = x_conv.numel();
    auto x_res = torch::empty_like(x_conv);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_tanh_hardswish_add_kernel<<<num_blocks, block_size>>>(
        x_conv.data_ptr<float>(),
        x_norm.data_ptr<float>(),
        x_res.data_ptr<float>(),
        size
    );
    
    return x_res;
}
"""

fused_tanh_hardswish_add_cpp_source = (
    "torch::Tensor fused_tanh_hardswish_add_cuda(torch::Tensor x_conv, torch::Tensor x_norm);"
)

fused_tanh_hardswish_add = load_inline(
    name="fused_tanh_hardswish_add",
    cpp_sources=fused_tanh_hardswish_add_cpp_source,
    cuda_sources=fused_tanh_hardswish_add_source,
    functions=["fused_tanh_hardswish_add_cuda"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses Tanh, HardSwish, and residual addition into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.fused_op = fused_tanh_hardswish_add

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        # Fused Tanh -> HardSwish -> Residual Addition
        x_res = self.fused_op.fused_tanh_hardswish_add_cuda(x_conv, x_norm)
        # LogSumExp
        x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
        return x_logsumexp

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda().contiguous()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]