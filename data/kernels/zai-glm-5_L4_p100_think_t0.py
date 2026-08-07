import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused clamp and division
fused_clamp_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_clamp_div_kernel(const float* __restrict__ input, float* __restrict__ output, int size, float min_val, float inv_divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = idx * 4; // Each thread processes 4 elements

    if (tid < size) {
        if (tid + 3 < size) {
            // Vectorized load/store using float4 for better memory throughput
            float4 val = reinterpret_cast<const float4*>(input)[idx];
            float4 res;
            res.x = fmaxf(val.x, min_val) * inv_divisor;
            res.y = fmaxf(val.y, min_val) * inv_divisor;
            res.z = fmaxf(val.z, min_val) * inv_divisor;
            res.w = fmaxf(val.w, min_val) * inv_divisor;
            reinterpret_cast<float4*>(output)[idx] = res;
        } else {
            // Handle remaining elements
            for (int i = 0; i < 4; ++i) {
                if (tid + i < size) {
                    float val = input[tid + i];
                    output[tid + i] = fmaxf(val, min_val) * inv_divisor;
                }
            }
        }
    }
}

torch::Tensor fused_clamp_div_cuda(torch::Tensor input, float min_val, float divisor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    float inv_divisor = 1.0f / divisor;

    const int block_size = 256;
    // Calculate grid size based on number of float4 elements
    int num_elements_4 = (size + 3) / 4;
    const int num_blocks = (num_elements_4 + block_size - 1) / block_size;

    fused_clamp_div_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, 
        min_val, 
        inv_divisor
    );

    return output;
}
"""

fused_clamp_div_cpp_source = (
    "torch::Tensor fused_clamp_div_cuda(torch::Tensor input, float min_val, float divisor);"
)

# Compile the inline CUDA code
fused_clamp_div = load_inline(
    name="fused_clamp_div",
    cpp_sources=fused_clamp_div_cpp_source,
    cuda_sources=fused_clamp_div_source,
    functions=["fused_clamp_div_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
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
        # Ensure memory is contiguous for vectorized access
        if not x.is_contiguous():
            x = x.contiguous()
        x = self.fused_clamp_div.fused_clamp_div_cuda(x, self.min_value, self.divisor)
        return x