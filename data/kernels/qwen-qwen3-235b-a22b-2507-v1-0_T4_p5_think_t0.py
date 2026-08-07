import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bias_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_tanh_kernel(const float* x, const float* bias, float* out, int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * height * width;
    if (idx >= total_elements) return;

    int spatial_size = height * width;
    int c = (idx / spatial_size) % channels;

    float val = x[idx] - bias[c];
    out[idx] = tanhf(val);
}

torch::Tensor fused_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
    TORCH_CHECK(bias.device().is_cuda(), "bias must be on CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be 4-dimensional");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3-dimensional");
    TORCH_CHECK(bias.size(0) == x.size(1) && bias.size(1) == 1 && bias.size(2) == 1, 
                "bias shape mismatch: expected (C,1,1) but got ", bias.sizes());

    int batch = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    auto out = torch::empty_like(x);

    int total_elements = batch * channels * height * width;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bias_tanh_kernel<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch, channels, height, width
    );

    return out;
}
"""

fused_bias_tanh_cpp = "torch::Tensor fused_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias);"

fused_op = load_inline(
    name="fused_bias_tanh",
    cpp_sources=fused_bias_tanh_cpp,
    cuda_sources=fused_bias_tanh_source,
    functions=["fused_bias_tanh_cuda"],
    verbose=False,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_bias_tanh_cuda(x, self.bias)
        return x