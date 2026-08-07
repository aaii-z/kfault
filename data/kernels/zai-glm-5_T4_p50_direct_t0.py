import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

// Kernel for fused scale, average pooling, bias addition, and scale
// Input x: (N, C, D, H, W)
// Output out: (N, C, D/2, H/2, W/2)
// Bias: (C, 1, 1, 1)
__global__ void fused_ops_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float scale1,
    float scale2,
    int N, int C, int D, int H, int W,
    int out_D, int out_H, int out_W) {

    // Calculate output dimensions
    int n = blockIdx.x; // Batch index
    int c = blockIdx.y; // Channel index
    
    // Calculate spatial index within the output feature map
    int spatial_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (spatial_idx >= out_D * out_H * out_W) return;

    int od = spatial_idx / (out_H * out_W);
    int rem = spatial_idx % (out_H * out_W);
    int oh = rem / out_W;
    int ow = rem % out_W;

    // Input pooling window coordinates
    int id_start = od * 2;
    int ih_start = oh * 2;
    int iw_start = ow * 2;

    // Calculate the starting index for the input tensor for this (n, c)
    // Input layout is NCDHW
    long long nc_offset = (long long)n * C * D * H * W + (long long)c * D * H * W;

    float sum_val = 0.0f;
    
    // Unroll the 2x2x2 pooling loop manually for efficiency
    // We assume D, H, W are even or padding handles boundaries, but here we assume standard valid pooling
    // Check boundaries just in case, though typical conv setups match dimensions
    
    // z=0, y=0, x=0
    long long in_idx = nc_offset + (long long)id_start * H * W + (long long)ih_start * W + iw_start;
    sum_val += x[in_idx];
    
    // z=0, y=0, x=1
    if (iw_start + 1 < W) sum_val += x[in_idx + 1];

    // z=0, y=1, x=0
    if (ih_start + 1 < H) {
        in_idx = nc_offset + (long long)id_start * H * W + (long long)(ih_start + 1) * W + iw_start;
        sum_val += x[in_idx];
        if (iw_start + 1 < W) sum_val += x[in_idx + 1];
    }

    // z=1
    if (id_start + 1 < D) {
        in_idx = nc_offset + (long long)(id_start + 1) * H * W + (long long)ih_start * W + iw_start;
        sum_val += x[in_idx];
        if (iw_start + 1 < W) sum_val += x[in_idx + 1];

        if (ih_start + 1 < H) {
            in_idx = nc_offset + (long long)(id_start + 1) * H * W + (long long)(ih_start + 1) * W + iw_start;
            sum_val += x[in_idx];
            if (iw_start + 1 < W) sum_val += x[in_idx + 1];
        }
    }

    // Average pool factor is 8 (2x2x2)
    float avg_val = sum_val * 0.125f; 

    // Apply scale1
    float scaled1_val = avg_val * scale1;

    // Add bias (broadcasted)
    float biased_val = scaled1_val + bias[c];

    // Apply scale2
    float result = biased_val * scale2;

    // Write to output
    // Output layout N C D/2 H/2 W/2
    long long out_idx = (long long)n * C * out_D * out_H * out_W + 
                        (long long)c * out_D * out_H * out_W + 
                        (long long)od * out_H * out_W + 
                        (long long)oh * out_W + 
                        ow;
    out[out_idx] = result;
}

torch::Tensor fused_ops_cuda(
    torch::Tensor x, 
    torch::Tensor bias, 
    float scale1, 
    float scale2) {
    
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    int out_D = D / 2;
    int out_H = H / 2;
    int out_W = W / 2;

    auto out = torch::zeros({N, C, out_D, out_H, out_W}, x.options());

    int spatial_size = out_D * out_H * out_W;
    int threads = 256;
    int spatial_blocks = (spatial_size + threads - 1) / threads;

    // Grid configuration:
    // dim3 block_x -> batch
    // dim3 block_y -> channel
    // dim3 block_z -> spatial blocks
    dim3 gridDim(N, C, spatial_blocks);
    dim3 blockDim(threads);

    fused_ops_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        scale1,
        scale2,
        N, C, D, H, W,
        out_D, out_H, out_W
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias, float scale1, float scale2);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution followed by a fused kernel
    for scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        # Use custom fused kernel
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.scale1.item(), self.scale2.item())
        return x