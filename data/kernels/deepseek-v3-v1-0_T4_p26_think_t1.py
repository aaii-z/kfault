import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel for element-wise addition and HardSwish activation
fused_add_activation_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish_activation(float x) {
    float x_plus_3 = x + 3.0f;
    float relu6 = fminf(fmaxf(x_plus_3, 0.0f), 6.0f);
    return x * relu6 / 6.0f;
}

__global__ void fused_add_activation_kernel(
    const float* conv_output,
    const float* add_input,
    float* output,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float conv_val = conv_output[idx];
        float add_val = add_input[idx];
        float tmp = conv_val + add_val;
        output[idx] = tmp * hardswish_activation(tmp);
    }
}

torch::Tensor fused_add_activation_cuda(
    torch::Tensor conv_output,
    torch::Tensor add_input
) {
    auto output = torch::empty_like(conv_output);
    int total_elements = conv_output.numel();
    
    const int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_add_activation_kernel<<<blocks_per_grid, threads_per_block>>>(
        conv_output.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements
    );
    
    return output;
}
"""

fused_add_activation_cpp_source = """
torch::Tensor fused_add_activation_cuda(torch::Tensor conv_output, torch::Tensor add_input);
"""

# Load the custom CUDA operator
fused_add_activation = load_inline(
    name="fused_add_activation",
    cpp_sources=fused_add_activation_cpp_source,
    cuda_sources=fused_add_activation_source,
    functions=["fused_add_activation_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding
        )

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        x = fused_add_activation.fused_add_activation_cuda(x, add_input)
        return x