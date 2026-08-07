import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_subtract_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_subtract_mish_kernel(const float* input, float* output, int size, float s_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - s_total;
        float sp = logf(1.0f + expf(x));
        float tanh_sp = tanhf(sp);
        output[idx] = x * tanh_sp;
    }
}

torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float s_total) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_subtract_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        s_total
    );

    return output;
}
"""

fused_subtract_mish_cpp_source = (
    "torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float s_total);"
)

fused_op = load_inline(
    name="fused_subtract_mish",
    cpp_sources=fused_subtract_mish_cpp_source,
    cuda_sources=fused_subtract_mish_source,
    functions=["fused_subtract_mish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses subtraction and Mish activation into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.s_total = subtract_value_1 + subtract_value_2
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_subtract_mish_cuda(x, self.s_total)
        return x