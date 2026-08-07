import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Clamp + Division
fused_clamp_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_clamp_div_kernel(
    const float* input,
    float* output,
    int num_elements,
    float min_value,
    float divisor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float value = input[idx];
        value = fmaxf(value, min_value);
        output[idx] = value / divisor;
    }
}

torch::Tensor fused_clamp_div_cuda(
    torch::Tensor input,
    float min_value,
    float divisor
) {
    CHECK_INPUT(input);
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_clamp_div_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        min_value,
        divisor
    );
    
    return output;
}
"""

fused_clamp_div_cpp_source = """
torch::Tensor fused_clamp_div_cuda(torch::Tensor input, float min_value, float divisor);
"""

# Compile the inline CUDA code
fused_clamp_div = load_inline(
    name="fused_clamp_div",
    cpp_sources=fused_clamp_div_cpp_source,
    cuda_sources=fused_clamp_div_source,
    functions=["fused_clamp_div_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_clamp_div.fused_clamp_div_cuda(x, self.min_value, self.divisor)
        return x