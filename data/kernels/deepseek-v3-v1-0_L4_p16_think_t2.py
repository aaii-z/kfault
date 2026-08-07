import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Mish + add + Hardtanh + scale
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float mish(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_ops_kernel(
    float* input_output,
    int num_elements,
    float add_value,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float value = input_output[idx];
        value = mish(value);
        value += add_value;
        value = fmaxf(-1.0f, fminf(1.0f, value));
        value *= scale;
        input_output[idx] = value;
    }
}

torch::Tensor fused_ops_forward(
    torch::Tensor input,
    float add_value,
    float scale
) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    
    auto output = input.clone();
    int num_elements = output.numel();
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_ops_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        num_elements,
        add_value,
        scale
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_ops_forward(torch::Tensor input, float add_value, float scale);
"""

# Load the custom CUDA extension
fused_ops_ext = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_forward"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ops_ext.fused_ops_forward(x, self.add_value, self.scale)
        return x