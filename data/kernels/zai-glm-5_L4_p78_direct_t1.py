import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused max pooling and sum reduction
# This kernel fuses the second max pool (kernel 3, stride 3) and the sum reduction along channels
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// Kernel for fused MaxPool3d (kernel 3, stride 3, padding 0) and Sum(dim=1)
// Input tensor shape: (N, C, D, H, W)
// Output tensor shape: (N, 1, D_out, H_out, W_out)
// D_out = floor((D - 3) / 3) + 1

__global__ void fused_maxpool_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int out_D, int out_H, int out_W) {

    // Calculate output spatial coordinates
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int od = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if this thread is within output spatial bounds
    if (ow < out_W && oh < out_H && od < out_D) {
        // Iterate over batch and channels
        for (int n = 0; n < N; ++n) {
            float sum_val = 0.0f;

            // Loop over channels
            for (int c = 0; c < C; ++c) {
                float max_val = -INFINITY;

                // Loop over pooling window (3x3x3)
                // Unrolled loops for constant size 3
                #pragma unroll
                for (int kd = 0; kd < 3; ++kd) {
                    #pragma unroll
                    for (int kh = 0; kh < 3; ++kh) {
                        #pragma unroll
                        for (int kw = 0; kw < 3; ++kw) {
                            int id = od * 3 + kd;
                            int ih = oh * 3 + kh;
                            int iw = ow * 3 + kw;

                            // Input index
                            long idx = ((long)n * C + c) * D * H * W + id * H * W + ih * W + iw;
                            float val = input[idx];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }
                sum_val += max_val;
            }
            
            // Write output
            // Output shape: (N, 1, out_D, out_H, out_W)
            long out_idx = ((long)n * out_D + od) * out_H * out_W + oh * out_W + ow;
            output[out_idx] = sum_val;
        }
    }
}

torch::Tensor fused_maxpool_sum_cuda(torch::Tensor input) {
    // Input dimensions
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    // Output dimensions for MaxPool3d with kernel=3, stride=3 (default)
    // floor((D - 3) / 3) + 1
    int out_D = (D - 3) / 3 + 1;
    int out_H = (H - 3) / 3 + 1;
    int out_W = (W - 3) / 3 + 1;

    auto output = torch::zeros({N, 1, out_D, out_H, out_W}, input.options());

    // Grid and Block dimensions
    dim3 block(8, 8, 4); // 256 threads
    dim3 grid((out_W + block.x - 1) / block.x,
              (out_H + block.y - 1) / block.y,
              (out_D + block.z - 1) / block.z);

    fused_maxpool_sum_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        out_D, out_H, out_W
    );

    return output;
}
"""

fused_cpp_source = "torch::Tensor fused_maxpool_sum_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_maxpool_sum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.fused_op.fused_maxpool_sum_cuda(x)
        return x