import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_along_channels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_along_channels_kernel(const float* input, float* output, int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * H * W;
    if (idx >= total_elements) return;

    int b = idx / (H * W);
    int remainder = idx % (H * W);
    int h = remainder / W;
    int w = remainder % W;

    int input_base = b * C * H * W + h * W + w;
    float min_val = input[input_base];

    for (int c = 1; c < C; c++) {
        int input_idx = input_base + c * H * W;
        float val = input[input_idx];
        if (val < min_val) min_val = val;
    }

    int output_idx = b * H * W + h * W + w;
    output[output_idx] = min_val;
}

torch::Tensor min_along_channels_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional (B, C, H, W)");
    x = x.contiguous();
    
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    auto output = torch::empty({B, 1, H, W}, x.options());
    int total_elements = B * H * W;
    
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    min_along_channels_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );
    
    return output;
}
"""

min_along_channels_cpp = "torch::Tensor min_along_channels_cuda(torch::Tensor x);"

min_along_channels = load_inline(
    name="min_along_channels",
    cpp_sources=min_along_channels_cpp,
    cuda_sources=min_along_channels_source,
    functions=["min_along_channels_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.min_along_channels = min_along_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.min_along_channels.min_along_channels_cuda(x)
        x = x * self.scale_factor
        return x