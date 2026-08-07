import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused MaxPool3d (k=2, s=2) + MaxPool3d (k=3, s=3) + Sum(dim=1)
# Mathematical analysis:
# MaxPool(k=2, s=2) followed by MaxPool(k=3, s=3) is equivalent to a single MaxPool(k=6, s=6).
# Input indices for the second pool are 3*i, 3*i+1, 3*i+2.
# Input indices for the first pool (relative to second pool input) are 2*j, 2*j+1.
# Combined, the input indices are 2*(3*i + k) + m = 6*i + 2*k + m.
# The range of 2*k + m for k in [0,2] and m in [0,1] is [0, 5].
# Thus, it is a contiguous window of size 6 with stride 6.
# We fuse this combined MaxPool with the subsequent Sum reduction over channels.

fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_maxpool_sum_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * D_out * H_out * W_out;

    if (idx < total_outputs) {
        // Decompose linear index into (n, d_out, h_out, w_out)
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int d_out = (idx / (W_out * H_out)) % D_out;
        int n = idx / (W_out * H_out * D_out);

        // Calculate input start indices for the 6x6x6 window
        int d_start = d_out * 6;
        int h_start = h_out * 6;
        int w_start = w_out * 6;

        // Strides for input tensor (N, C, D, H, W)
        // Assuming contiguous memory layout
        long long stride_n = (long long)C * D * H * W;
        long long stride_c = (long long)D * H * W;
        long long stride_d = (long long)H * W;
        long long stride_h = W;
        long long stride_w = 1;

        const float* ptr_n = input + n * stride_n;

        float sum_val = 0.0f;

        // Loop over channels
        for (int c = 0; c < C; ++c) {
            float max_val = -INFINITY;

            // Loop over the 6x6x6 window
            // Unrolling might be beneficial, but compiler usually handles constant loops well.
            // We iterate relative to the start of the window
            const float* ptr_c = ptr_n + c * stride_c;

            for (int kd = 0; kd < 6; ++kd) {
                for (int kh = 0; kh < 6; ++kh) {
                    // Base pointer for the row (d, h)
                    const float* row_start = ptr_c + (d_start + kd) * stride_d + (h_start + kh) * stride_h + w_start * stride_w;
                    
                    // Iterate over width (contiguous in memory)
                    for (int kw = 0; kw < 6; ++kw) {
                        float val = row_start[kw];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
            }
            sum_val += max_val;
        }
        output[idx] = sum_val;
    }
}

torch::Tensor fused_maxpool_sum_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    // Output dimensions for the fused operation
    // Equivalent to MaxPool3d(kernel=6, stride=6)
    int D_out = (D - 6) / 6 + 1;
    int H_out = (H - 6) / 6 + 1;
    int W_out = (W - 6) / 6 + 1;

    auto output = torch::zeros({N, 1, D_out, H_out, W_out}, input.options());

    int total_outputs = N * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;

    fused_maxpool_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N, C, D, H, W, 
        D_out, H_out, W_out
    );

    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_maxpool_sum_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_maxpool_sum",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_maxpool_sum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution followed by a fused
    MaxPool + Sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        # Replace max_pool1, max_pool2, and sum with the fused CUDA kernel
        x = self.fused_op.fused_maxpool_sum_cuda(x)
        return x