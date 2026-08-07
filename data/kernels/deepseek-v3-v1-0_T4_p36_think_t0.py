import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define BLOCK_SIZE 256

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / 1.41421356237f));
}

__global__ void fused_ops_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    
    if (col >= width || batch >= batch_size) return;

    float min_val = 3.402823466e+38F; // FLT_MAX
    float sum_val = 0.0f;

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            int idx = ((batch * channels + c) * height + h) * width + col;
            float val = input[idx];
            if (val < min_val) min_val = val;
        }
        sum_val += min_val;
        min_val = 3.402823466e+38F;
    }

    float result = gelu(sum_val) + bias[0];
    int out_idx = batch * width + col;
    output[out_idx] = result;
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::empty({batch_size, 1, 1, width}, input.options());
    
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    dim3 threads(BLOCK_SIZE);
    
    fused_ops_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias)
        return x