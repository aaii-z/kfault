import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_clamp_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_clamp_div_kernel(const float* input, float* output, int size, float min_value, float inv_divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        float clamped = (val < min_value) ? min_value : val;
        output[idx] = clamped * inv_divisor;
    }
}

torch::Tensor fused_clamp_div_cuda(torch::Tensor input, float min_value, float divisor) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input must be float32");
    
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const float inv_divisor = 1.0f / divisor;

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_clamp_div_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, 
        min_value, 
        inv_divisor
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

fused_clamp_div_cpp_source = "torch::Tensor fused_clamp_div_cuda(torch::Tensor input, float min_value, float divisor);"

fused_clamp_div = load_inline(
    name="fused_clamp_div",
    cpp_sources=fused_clamp_div_cpp_source,
    cuda_sources=fused_clamp_div_source,
    functions=["fused_clamp_div_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.fused_clamp_div = fused_clamp_div

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_clamp_div.fused_clamp_div_cuda(x, self.min_value, self.divisor)
        return x