import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused min + sum + gelu + bias
fused_postprocess_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation approximation: x * 0.5 * (1 + erf(x * 0.7071067811865476))
__device__ float gelu(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
}

__global__ void fused_min_sum_gelu_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * W;
    if (idx >= total_elements) return;

    int b = idx / W;
    int w = idx % W;

    float sum_val = 0.0f;
    for (int h = 0; h < H; h++) {
        float min_val = input[b * C * H * W + 0 * H * W + h * W + w];
        for (int c = 1; c < C; c++) {
            float val = input[b * C * H * W + c * H * W + h * W + w];
            min_val = fminf(min_val, val);
        }
        sum_val += min_val;
    }

    float result = gelu(sum_val) + bias[0];
    output[b * 1 * 1 * W + 0 * 1 * W + 0 * W + w] = result;
}
"""

fused_postprocess_cpp = """
torch::Tensor fused_min_sum_gelu_bias_launcher(
    torch::Tensor input,
    torch::Tensor bias,
    int B, int C, int H, int W);
"""

fused_postprocess_pybind_wrapper = """
torch::Tensor fused_min_sum_gelu_bias_launcher(
    torch::Tensor input,
    torch::Tensor bias,
    int B, int C, int H, int W) {
    
    auto output = torch::empty({B, 1, 1, W}, input.options());

    const int block_size = 256;
    const int num_blocks = (B * W + block_size - 1) / block_size;

    fused_min_sum_gelu_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );

    return output;
}
"""

# Compile the fused post-processing kernel
fused_ops = load_inline(
    name="fused_postprocess",
    cpp_sources=[fused_postprocess_cpp, fused_postprocess_pybind_wrapper],
    cuda_sources=[fused_postprocess_source],
    functions=["fused_min_sum_gelu_bias_launcher"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for min + sum + gelu + bias.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_kernel = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)  # [B, C_out, H_out, W_out]

        B, C, H, W = x.shape

        input_contig = x.contiguous()

        output = self.fused_kernel.fused_min_sum_gelu_bias_launcher(input_contig, self.bias, B, C, H, W)

        return output