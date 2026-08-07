import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bias_scale_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_scale_sigmoid_kernel(
    const float* softmax_output,
    const float* bias,
    float* output,
    float scaling_factor,
    int B,
    int C,
    int H,
    int W) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * H * W;
    if (idx >= total_elements) return;

    int spatial_size = H * W;
    int c = (idx % (C * spatial_size)) / spatial_size;

    float val = softmax_output[idx] + bias[c];
    val = val * scaling_factor;
    output[idx] = 1.0f / (1.0f + expf(-val));
}

torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor softmax_output,
    torch::Tensor bias,
    float scaling_factor) {

    TORCH_CHECK(softmax_output.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
    TORCH_CHECK(softmax_output.dim() == 4, "Input tensor must be 4D (B, C, H, W)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

    softmax_output = softmax_output.contiguous();
    bias = bias.contiguous();

    auto B = softmax_output.size(0);
    auto C = softmax_output.size(1);
    auto H = softmax_output.size(2);
    auto W = softmax_output.size(3);
    auto bias_size = bias.size(0);

    TORCH_CHECK(C == bias_size, "Bias size must match number of channels");

    auto output = torch::empty_like(softmax_output);
    int total_elements = B * C * H * W;

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        softmax_output.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        B, C, H, W
    );

    return output;
}
"""

fused_bias_scale_sigmoid_cpp = "torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor, torch::Tensor, float);"

fused_bias_scale_sigmoid = load_inline(
    name="fused_bias_scale_sigmoid",
    cpp_sources=fused_bias_scale_sigmoid_cpp,
    cuda_sources=fused_bias_scale_sigmoid_source,
    functions=["fused_bias_scale_sigmoid_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses bias addition, scaling, and sigmoid into a single CUDA kernel after softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_bias_scale_sigmoid

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.softmax(x, dim=1)
        x = self.fused_op.fused_bias_scale_sigmoid_cuda(x, self.bias.view(-1), self.scaling_factor)
        return x