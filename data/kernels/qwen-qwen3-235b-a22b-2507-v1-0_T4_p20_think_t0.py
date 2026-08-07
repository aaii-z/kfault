import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fusing bias add, residual adds, and multiply operations
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel(const float* conv_out, const float* bias, float* out, 
                               int B, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * D * H * W;
    if (idx >= total_elements) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int c = (idx / (W * H * D)) % C;
    int b = idx / (W * H * D * C);

    float conv_val = conv_out[idx];
    float bias_val = bias[c];

    float result = conv_val * (2.0f * conv_val + bias_val + 1.0f);
    out[idx] = result;
}

torch::Tensor fused_op_cuda(torch::Tensor conv_out, torch::Tensor bias) {
    auto B = conv_out.size(0);
    auto C = conv_out.size(1);
    auto D = conv_out.size(2);
    auto H = conv_out.size(3);
    auto W = conv_out.size(4);
    auto total_elements = B * C * D * H * W;

    auto out = torch::empty_like(conv_out);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_op_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, D, H, W
    );

    return out;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor conv_out, torch::Tensor bias);"

# Compile the inline CUDA code for the fused operation
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel to fuse the operations after convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_op_cuda(x, self.bias)
        return x