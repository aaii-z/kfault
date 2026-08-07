import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: Scale -> AvgPool -> Bias -> Scale
# This kernel fuses the following PyTorch operations:
# 1. x = x * scale1
# 2. x = avg_pool3d(x, kernel_size=2)
# 3. x = x + bias
# 4. x = x * scale2
# Fusing these operations reduces global memory access significantly.

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    const float* __restrict__ scale1_ptr,
    const float* __restrict__ scale2_ptr,
    int N, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int stride_in_N, int stride_in_C, int stride_in_D, int stride_in_H, int stride_in_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D_out * H_out * W_out;

    if (idx < total_elements) {
        // Load scales (assumes scalar tensors)
        float scale1 = *scale1_ptr;
        float scale2 = *scale2_ptr;

        // Decode linear index to output coordinates (N, C, D_out, H_out, W_out)
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int d_out = (idx / (W_out * H_out)) % D_out;
        int c = (idx / (W_out * H_out * D_out)) % C;
        int n = idx / (W_out * H_out * D_out * C);

        // Calculate input start coordinates for the 2x2x2 pooling window
        int d_in = d_out * 2;
        int h_in = h_out * 2;
        int w_in = w_out * 2;

        float sum = 0.0f;

        // Calculate base pointer offset for the start of the window
        // Using strides to handle any memory layout (though ConvTranspose3d output is contiguous)
        int base_offset = n * stride_in_N + c * stride_in_C + d_in * stride_in_D + h_in * stride_in_H + w_in * stride_in_W;

        // Unrolled loop for the 2x2x2 window
        // Window element (0,0,0)
        sum += input[base_offset + 0 * stride_in_D + 0 * stride_in_H + 0 * stride_in_W];
        // Window element (0,0,1)
        sum += input[base_offset + 0 * stride_in_D + 0 * stride_in_H + 1 * stride_in_W];
        // Window element (0,1,0)
        sum += input[base_offset + 0 * stride_in_D + 1 * stride_in_H + 0 * stride_in_W];
        // Window element (0,1,1)
        sum += input[base_offset + 0 * stride_in_D + 1 * stride_in_H + 1 * stride_in_W];
        // Window element (1,0,0)
        sum += input[base_offset + 1 * stride_in_D + 0 * stride_in_H + 0 * stride_in_W];
        // Window element (1,0,1)
        sum += input[base_offset + 1 * stride_in_D + 0 * stride_in_H + 1 * stride_in_W];
        // Window element (1,1,0)
        sum += input[base_offset + 1 * stride_in_D + 1 * stride_in_H + 0 * stride_in_W];
        // Window element (1,1,1)
        sum += input[base_offset + 1 * stride_in_D + 1 * stride_in_H + 1 * stride_in_W];

        // Average Pool
        float avg = sum / 8.0f;

        // Apply Scale1, Bias, Scale2
        // bias is shape (C, 1, 1, 1), so we index with 'c'
        float b = bias[c];
        float res = (avg * scale1 + b) * scale2;

        output[idx] = res;
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor scale1,
    torch::Tensor scale2
) {
    // Input shape: (N, C, D_in, H_in, W_in)
    auto N = input.size(0);
    auto C = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);

    // Output shape calculation for AvgPool3d with kernel_size=2
    // D_out = floor(D_in / 2), etc.
    auto D_out = D_in / 2;
    auto H_out = H_in / 2;
    auto W_out = W_in / 2;

    auto output = torch::zeros({N, C, D_out, H_out, W_out}, input.options());

    int total_elements = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Get strides of input tensor
    auto stride_in = input.strides();

    fused_op_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale1.data_ptr<float>(),
        scale2.data_ptr<float>(),
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride_in[0], stride_in[1], stride_in[2], stride_in[3], stride_in[4]
    );

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale1, torch::Tensor scale2);"
)

# Compile the inline CUDA code
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
    Optimized Model that performs a 3D transposed convolution followed by a fused
    (Scale -> AvgPool -> Bias -> Scale) CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        # Apply the fused kernel
        x = self.fused_op.fused_op_cuda(x, self.bias, self.scale1, self.scale2)
        return x