import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_hardswish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float y = (x > 0.0f) ? x : 0.0f;
        float temp = (y + 3.0f) / 6.0f;
        float clamped_temp = (temp < 1.0f) ? temp : 1.0f;
        output[idx] = y * clamped_temp;
    }
}

torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_relu_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    return output;
}
"""

fused_activation_cpp = "torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input);"

fused_relu_hardswish = load_inline(
    name="fused_relu_hardswish",
    cpp_sources=fused_activation_cpp,
    cuda_sources=fused_activation_source,
    functions=["fused_relu_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses ReLU and HardSwish activations into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_activation = fused_relu_hardswish

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_relu_hardswish_cuda(x)
        return x