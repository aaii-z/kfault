import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_kernel(const float* x, const float* scaling_factor, const float* bias, float* out, 
                            int total_elements, int depth, int height, int width, int out_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int stride_channel = depth * height * width;
    int c = (idx / stride_channel) % out_channels;

    float val = x[idx] * scaling_factor[c];
    val = tanhf(val);
    val = val * bias[c];
    val = 1.0f / (1.0f + expf(-val));

    out[idx] = val;
}

torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor scaling_factor, torch::Tensor bias) {
    auto out = torch::empty_like(x);
    int total_elements = x.numel();
    int out_channels = x.size(1);
    int depth = x.size(2);
    int height = x.size(3);
    int width = x.size(4);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        scaling_factor.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements,
        depth, height, width, out_channels
    );

    return out;
}
"""

fused_op_cpp = "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor scaling_factor, torch::Tensor bias);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_op_cuda(x, self.scaling_factor, self.bias)
        return x