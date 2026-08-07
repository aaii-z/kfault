import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_sub_tanh_sub_kernel(
    const float* input,
    float* output,
    const float sub1_val,
    const float sub2_val,
    int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx] - sub1_val;
        val = tanhf(val);
        output[idx] = val - sub2_val;
    }
}

torch::Tensor fused_sub_tanh_sub_cuda(
    torch::Tensor input,
    float sub1_val,
    float sub2_val
) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be FP32");

    auto output = torch::empty_like(input);
    const int num_elements = input.numel();
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    fused_sub_tanh_sub_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        sub1_val,
        sub2_val,
        num_elements
    );

    return output;
}
"""

fused_ops_cpp = "torch::Tensor fused_sub_tanh_sub_cuda(torch::Tensor, float, float);"

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_ops_source,
    functions=["fused_sub_tanh_sub_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_sub_tanh_sub_cuda(x, self.subtract1_value, self.subtract2_value)
        x = self.avgpool(x)
        return x