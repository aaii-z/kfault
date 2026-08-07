import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for subtract and mish operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float mish_activation(float x) {
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else {
        sp = log1pf(expf(x));
    }
    return x * tanhf(sp);
}

__global__ void fused_subtract_mish_kernel(
    const float* input,
    float* output,
    float subtract_value_1,
    float subtract_value_2,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = input[idx] - subtract_value_1 - subtract_value_2;
        output[idx] = mish_activation(val);
    }
}

torch::Tensor fused_subtract_mish_cuda(
    torch::Tensor input,
    float subtract_value_1,
    float subtract_value_2
) {
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_subtract_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        subtract_value_1,
        subtract_value_2,
        total_elements
    );
    
    return output;
}
"""

fused_kernel_cpp = "torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float subtract_value_1, float subtract_value_2);"

# Load the custom CUDA extension
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_subtract_mish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_subtract_mish_cuda(x, self.subtract_value_1, self.subtract_value_2)
        return x