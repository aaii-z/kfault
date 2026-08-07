import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused pointwise operations
fused_op_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/util/Half.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__global__ void fused_op_kernel(
    const float* input,
    float* output,
    float add_value,
    float multiply_value,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx] + add_value;
        val = min(val, 0.0f);
        val = gelu(val);
        output[idx] = val * multiply_value;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    float add_value,
    float multiply_value
) {
    CHECK_INPUT(input);
    auto output = torch::empty_like(input);
    int num_elements = output.numel();
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    fused_op_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        add_value,
        multiply_value,
        num_elements
    );
    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, float add_value, float multiply_value);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_op.fused_op_cuda(x, self.add_value, self.multiply_value)
        return x