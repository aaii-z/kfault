import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + Scale + Min along channel dimension with high parallelism
fused_conv_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

#define TILE_SIZE 16
#define KERNEL_RADIUS 1

// Shared memory for input tile
__shared__ float input_shared[TILE_SIZE + 2 * KERNEL_RADIUS][TILE_SIZE + 2 * KERNEL_RADIUS];

// Fused kernel: Conv2d (with 3x3 kernel, padding=1) + scale + channel-wise min
__global__ void fused_conv_scale_min_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    float scale_factor
) {
    int batch = blockIdx.x;
    int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
    int tx = blockIdx.z * TILE_SIZE + threadIdx.x;

    int tiled_h = (height + TILE_SIZE - 1) / TILE_SIZE;
    int tiled_w = (width + TILE_SIZE - 1) / TILE_SIZE;

    // Thread indices within the tile
    int ty_local = threadIdx.y;
    int tx_local = threadIdx.x;

    // Global output position
    if (ty >= height || tx >= width) return;

    float min_val = FLT_MAX;

    // Loop over input channels in batches of one (assumes in_channels is manageable)
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load input tile into shared memory with halo
        int iy_start = ty - KERNEL_RADIUS;
        int ix_start = tx - KERNEL_RADIUS;

        if (ty_local == 0) {
            // Load top halo row
            int iy = iy_start - 1;
            int src_idx = batch * in_channels * height * width + ic * height * width + max(iy, 0) * width + max(ix_start, 0);
            int dst_idx = (-1) * (TILE_SIZE + 2 * KERNEL_RADIUS) + tx_local;
            input_shared[-1][tx_local] = (iy < 0 || ix_start < 0) ? 0.0f : input[src_idx];
        }
        if (ty_local == TILE_SIZE - 1) {
            // Load bottom halo row
            int iy = ty + KERNEL_RADIUS;
            int src_idx = batch * in_channels * height * width + ic * height * width + min(iy, height - 1) * width + max(ix_start, 0);
            int dst_idx = TILE_SIZE * (TILE_SIZE + 2 * KERNEL_RADIUS) + tx_local;
            input_shared[TILE_SIZE + 1][tx_local] = (iy >= height || ix_start < 0) ? 0.0f : input[src_idx];
        }
        if (tx_local == 0) {
            // Load left halo column
            int ix = ix_start - 1;
            int src_idx = batch * in_channels * height * width + ic * height * width + max(iy_start, 0) * width + max(ix, 0);
            int dst_idx = ty_local * (TILE_SIZE + 2 * KERNEL_RADIUS) + (-1);
            input_shared[ty_local][(-1)] = (ix < 0 || iy_start < 0) ? 0.0f : input[src_idx];
        }
        if (tx_local == TILE_SIZE - 1) {
            // Load right halo column
            int ix = tx + KERNEL_RADIUS;
            int src_idx = batch * in_channels * height * width + ic * height * width + max(iy_start, 0) * width + min(ix, width - 1);
            int dst_idx = ty_local * (TILE_SIZE + 2 * KERNEL_RADIUS) + TILE_SIZE + 1;
            input_shared[ty_local][TILE_SIZE + 1] = (ix >= width || iy_start < 0) ? 0.0f : input[src_idx];
        }

        // Center load
        int src_idx = batch * in_channels * height * width + ic * height * width + iy_start * width + ix_start;
        input_shared[ty_local][tx_local] = (iy_start < 0 || iy_start >= height || ix_start < 0 || ix_start >= width) ? 0.0f : input[src_idx];

        __syncthreads();

        // Compute conv for all output channels
        #pragma unroll
        for (int oc = 0; oc < out_channels; ++oc) {
            float conv_val = bias[oc];

            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    float input_val = input_shared[ty_local + ky][tx_local + kx];
                    int w_idx = oc * in_channels * 9 + ic * 9 + ky * 3 + kx;
                    conv_val += input_val * weight[w_idx];
                }
            }

            conv_val *= scale_factor;
            min_val = fminf(min_val, conv_val);
        }

        __syncthreads();
    }

    // Write output: (batch, 1, height, width)
    int output_idx = batch * height * width + ty * width + tx;
    output[output_idx] = min_val;
}
"""

fused_conv_scale_min_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor);
"""

# Wrapper that launches the kernel
fused_conv_scale_min_wrapper = """
torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor) {
    
    auto batch_size = input.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);

    auto output = torch::empty({batch_size, 1, height, width}, input.options());

    const int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid(batch_size,
              (height + TILE_SIZE - 1) / TILE_SIZE,
              (width + TILE_SIZE - 1) / TILE_SIZE);

    fused_conv_scale_min_kernel<<<grid, dim3(TILE_SIZE, TILE_SIZE), 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input.size(1),
        out_channels,
        height,
        width,
        scale_factor
    );

    return output;
}
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_scale_min",
    cpp_sources=[fused_conv_scale_min_cpp_source, fused_conv_scale_min_wrapper],
    cuda_sources=fused_conv_scale_min_source,
    functions=["fused_conv_scale_min_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + Scale + Min along channel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Initialize weights and bias as in Conv2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

        # Fused CUDA function
        self.fused_op = fused_op.fused_conv_scale_min_cuda

    def forward(self, x):
        return self.fused_op(x, self.weight, self.bias, self.scale_factor)