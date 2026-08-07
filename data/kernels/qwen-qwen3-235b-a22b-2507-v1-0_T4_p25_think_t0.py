import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_along_channels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void min_along_channels_kernel(const float* input, float* output, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W;
    if (idx >= total) return;

    int b = idx / (H * W);
    int hw = idx % (H * W);
    int h = hw / W;
    int w = hw % W;

    float min_val = FLT_MAX;
    for (int c = 0; c < C; c++) {
        int input_idx = b * C * H * W + c * H * W + h * W + w;
        float val = input[input_idx];
        if (val < min_val) min_val = val;
    }
    int output_idx = b * H * W + h * W + w;
    output[output_idx] = min_val;
}

torch::Tensor min_along_channels_cuda(torch::Tensor input) {
    auto B = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::empty({B, 1, H, W}, input.options());
    
    int total_elements = B * H * W;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    min_along_channels_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );
    
    return output;
}
"""

min_along_channels_cpp = "torch::Tensor min_along_channels_cuda(torch::Tensor input);"

min_op = load_inline(
    name="min_along_channels",
    cpp_sources=min_along_channels_cpp,
    cuda_sources=min_along_channels_source,
    functions=["min_along_channels_cuda"],
    verbose=False
)

tanh_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void tanh_tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float y = tanhf(x);
        output[idx] = tanhf(y);
    }
}

torch::Tensor tanh_tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    tanh_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

tanh_tanh_cpp = "torch::Tensor tanh_tanh_cuda(torch::Tensor input);"

tanh_tanh_op = load_inline(
    name="tanh_tanh",
    cpp_sources=tanh_tanh_cpp,
    cuda_sources=tanh_tanh_source,
    functions=["tanh_tanh_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators for min reduction and fused tanh operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.min_op = min_op
        self.tanh_tanh_op = tanh_tanh_op

    def forward(self, x):
        x = self.conv(x)
        x = self.min_op.min_along_channels_cuda(x)
        x = self.tanh_tanh_op.tanh_tanh_cuda(x)
        return x