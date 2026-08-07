import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + Scale + Min along channel dimension with proper parallelism
fused_conv_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

#define TILE_SIZE 16
#define KERNEL_RADIUS 1

// Shared memory for input tile
__shared__ float shared_input[TILE_SIZE + 2 * KERNEL_RADIUS][TILE_SIZE + 2 * KERNEL_RADIUS];

// Fused kernel: Conv2d (3x3) + Scale + Channel-wise Min
__global__ void fused_conv_scale_min_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int H,
    int W,
    float scale_factor
) {
    int batch = blockIdx.x;
    int by = blockIdx.y;
    int bx = blockIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int oh = by * TILE_SIZE + ty;
    int ow = bx * TILE_SIZE + tx;

    // Offset for current batch
    int input_batch_offset = batch * in_channels * H * W;
    int output_batch_offset = batch * H * W;

    // Load halo tile into shared memory (including padding)
    int halo_h = TILE_SIZE + 2 * KERNEL_RADIUS;
    int halo_w = TILE_SIZE + 2 * KERNEL_RADIUS;
    int load_h = by * TILE_SIZE + ty - KERNEL_RADIUS;
    int load_w = bx * TILE_SIZE + tx - KERNEL_RADIUS;

    float input_val = 0.0f;
    if (load_h >= 0 && load_h < H && load_w >= 0 && load_w < W) {
        input_val = input[input_batch_offset + load_h * W + load_w];
    }
    shared_input[ty][tx] = input_val;
    __syncthreads();

    // Each thread computes one output element (if in bounds)
    if (oh >= H || ow >= W) return;

    float min_val = FLT_MAX;

    #pragma unroll
    for (int oc = 0; oc < out_channels; ++oc) {
        float conv_val = bias[oc];

        #pragma unroll
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* filter_ptr = weight + oc * in_channels * 9 + ic * 9;
            float* input_tile_ptr = &shared_input[ty][tx];

            // 3x3 convolution on shared tile
            conv_val += filter_ptr[0] * input_tile_ptr[-2 - TILE_SIZE];
            conv_val += filter_ptr[1] * input_tile_ptr[-2 - 0];
            conv_val += filter_ptr[2] * input_tile_ptr[-2 + TILE_SIZE];
            conv_val += filter_ptr[3] * input_tile_ptr[-1 - TILE_SIZE];
            conv_val += filter_ptr[4] * input_tile_ptr[-1 - 0];
            conv_val += filter_ptr[5] * input_tile_ptr[-1 + TILE_SIZE];
            conv_val += filter_ptr[6] * input_tile_ptr[ 0 - TILE_SIZE];
            conv_val += filter_ptr[7] * input_tile_ptr[ 0 - 0];
            conv_val += filter_ptr[8] * input_tile_ptr[ 0 + TILE_SIZE];
        }

        conv_val *= scale_factor;
        min_val = fminf(min_val, conv_val);
    }

    output[output_batch_offset + oh * W + ow] = min_val;
}
"""

fused_conv_scale_min_cpp_source = """
#include <torch/extension.h>
std::vector<torch::Tensor> fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor);
"""

fused_op = load_inline(
    name="fused_conv_scale_min",
    cpp_sources=fused_conv_scale_min_cpp_source,
    cuda_sources=fused_conv_scale_min_source,
    functions=["fused_conv_scale_min_kernel"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + Scale + Min along channel.
    Uses tiling, shared memory, and proper grid-stride to maximize throughput.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        assert kernel_size == 3, "Only 3x3 kernels supported in fused kernel"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

        self.fused_op = fused_op

    def forward(self, x):
        batch_size, _, H, W = x.shape
        out_channels = self.out_channels

        # Launch configuration
        tile_size = 16
        grid_dim_x = batch_size
        grid_dim_y = (H + tile_size - 1) // tile_size
        grid_dim_z = (W + tile_size - 1) // tile_size
        block_dim = (tile_size, tile_size, 1)

        # Output: (batch, H, W)
        output = torch.empty((batch_size, H, W), device=x.device, dtype=x.dtype)

        self.fused_op.fused_conv_scale_min_kernel(
            x,
            self.weight,
            self.bias,
            output,
            batch_size,
            self.in_channels,
            out_channels,
            H,
            W,
            self.scale_factor,
            grid=(grid_dim_x, grid_dim_y, grid_dim_z),
            block=block_dim
        )

        return output.unsqueeze(1)  # (batch, 1, H, W)