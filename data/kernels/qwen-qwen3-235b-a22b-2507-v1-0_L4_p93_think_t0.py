import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>

#define M_SQRT_2_PI 0.79788456080286535588f

__global__ void fused_kernel(const float* input, float* output, float add_value, float multiply_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] + add_value;
        if (val > 0.0f) {
            val = 0.0f;
        }
        float x3 = val * val * val;
        float inner = M_SQRT_2_PI * (val + 0.044715f * x3);
        float tanh_inner = tanhf(inner);
        float gelu_val = 0.5f * val * (1.0f + tanh_inner);
        gelu_val = gelu_val * multiply_value;
        output[idx] = gelu_val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, float add_value, float multiply_value) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    auto output = torch::empty_like(input);
    int size = input.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        add_value,
        multiply_value,
        size
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
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_op_cuda(x, self.add_value, self.multiply_value)
        return x