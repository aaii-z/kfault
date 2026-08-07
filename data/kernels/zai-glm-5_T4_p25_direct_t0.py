import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min, tanh, tanh
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_ops_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int out_channels, int height, int width) {
    // Calculate global thread index
    // We map each thread to one spatial pixel in the output (batch, height, width)
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx = blockIdx.z;

    if (w_idx < width && h_idx < height && b_idx < batch_size) {
        // Calculate the starting index for the input channels at this spatial location
        // Input layout: (N, C, H, W)
        int spatial_offset = b_idx * out_channels * height * width + h_idx * width + w_idx;
        
        // Initialize min_val with the first channel
        // We iterate through the channels to find the minimum
        float min_val = input[spatial_offset];
        
        for (int c = 1; c < out_channels; ++c) {
            float val = input[spatial_offset + c * height * width];
            if (val < min_val) {
                min_val = val;
            }
        }
        
        // Apply tanh twice
        float t1 = tanhf(min_val);
        float t2 = tanhf(t1);
        
        // Write to output
        // Output layout: (N, 1, H, W)
        // output index: b_idx * height * width + h_idx * width + w_idx
        output[b_idx * height * width + h_idx * width + w_idx] = t2;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input) {
    // Input dimensions: (batch_size, out_channels, height, width)
    int batch_size = input.size(0);
    int out_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    // Output dimensions: (batch_size, 1, height, width)
    auto output = torch::zeros({batch_size, 1, height, width}, input.options());

    // Block and grid dimensions
    dim3 block(16, 16); // 256 threads per block
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y, 
              batch_size);

    fused_ops_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_channels, height, width);

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution followed by a fused CUDA kernel
    for minimum operation and double Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_ops_cuda(x)
        return x