import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sum_channels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_channels_kernel(const float* x, float* out, int B, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * D * H * W;
    if (idx >= total_elements) return;

    int b = idx / (D * H * W);
    int remainder = idx % (D * H * W);
    int d = remainder / (H * W);
    int remainder2 = remainder % (H * W);
    int h = remainder2 / W;
    int w = remainder2 % W;

    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        int x_index = (((b * C + c) * D + d) * H + h) * W + w;
        sum += x[x_index];
    }
    int out_index = b * (D * H * W) + (d * H * W + h * W + w);
    out[out_index] = sum;
}

torch::Tensor sum_channels_cuda(torch::Tensor x) {
    auto B = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);

    auto out = torch::zeros({B, 1, D, H, W}, x.options());

    int total_elements = B * D * H * W;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    sum_channels_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        B, C, D, H, W
    );

    return out;
}
"""

sum_channels_cpp = "torch::Tensor sum_channels_cuda(torch::Tensor x);"

sum_channels = load_inline(
    name="sum_channels",
    cpp_sources=sum_channels_cpp,
    cuda_sources=sum_channels_source,
    functions=["sum_channels_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)
        self.sum_channels = sum_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = self.sum_channels.sum_channels_cuda(x)
        return x