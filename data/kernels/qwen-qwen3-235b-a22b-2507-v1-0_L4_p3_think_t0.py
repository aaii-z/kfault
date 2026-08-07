import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom fused AvgPool3d + GELU CUDA kernel
fused_avgpool_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + coeff * x3)));
}

__global__ void fused_avgpool_gelu_kernel(
    const float* input,
    float* output,
    int batch,
    int channels,
    int input_D,
    int input_H,
    int input_W,
    int output_D,
    int output_H,
    int output_W,
    int pool_size_D,
    int pool_size_H,
    int pool_size_W
) {
    int total_elements = batch * channels * output_D * output_H * output_W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int w_out = idx % output_W;
    int h_out = (idx / output_W) % output_H;
    int d_out = (idx / (output_W * output_H)) % output_D;
    int c = (idx / (output_W * output_H * output_D)) % channels;
    int b = idx / (output_W * output_H * output_D * channels);

    int d_in = d_out * pool_size_D;
    int h_in = h_out * pool_size_H;
    int w_in = w_out * pool_size_W;

    float sum = 0.0f;
    for (int dz = 0; dz < pool_size_D; dz++) {
        for (int hy = 0; hy < pool_size_H; hy++) {
            for (int wx = 0; wx < pool_size_W; wx++) {
                int d = d_in + dz;
                int h = h_in + hy;
                int w = w_in + wx;
                int input_idx = (((b * channels + c) * input_D + d) * input_H + h) * input_W + w;
                sum += input[input_idx];
            }
        }
    }
    
    float avg = sum / (pool_size_D * pool_size_H * pool_size_W);
    float result = gelu(avg);
    
    int output_idx = (((b * channels + c) * output_D + d_out) * output_H + h_out) * output_W + w_out;
    output[output_idx] = result;
}

torch::Tensor fused_avgpool_gelu_cuda(
    torch::Tensor input,
    int pool_size_D,
    int pool_size_H,
    int pool_size_W
) {
    auto input_sizes = input.sizes();
    int batch = input_sizes[0];
    int channels = input_sizes[1];
    int input_D = input_sizes[2];
    int input_H = input_sizes[3];
    int input_W = input_sizes[4];
    
    int output_D = input_D / pool_size_D;
    int output_H = input_H / pool_size_H;
    int output_W = input_W / pool_size_W;
    
    auto output = torch::empty({batch, channels, output_D, output_H, output_W}, input.options());
    
    int total_elements = batch * channels * output_D * output_H * output_W;
    int block_size = BLOCK_SIZE;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_avgpool_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        input_D,
        input_H,
        input_W,
        output_D,
        output_H,
        output_W,
        pool_size_D,
        pool_size_H,
        pool_size_W
    );
    
    return output;
}
"""

fused_avgpool_gelu_cpp = "torch::Tensor fused_avgpool_gelu_cuda(torch::Tensor input, int pool_size_D, int pool_size_H, int pool_size_W);"

fused_avgpool_gelu = load_inline(
    name="fused_avgpool_gelu",
    cpp_sources=fused_avgpool_gelu_cpp,
    cuda_sources=fused_avgpool_gelu_source,
    functions=["fused_avgpool_gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.pool_kernel_size = pool_kernel_size
        self.fused_avgpool_gelu = fused_avgpool_gelu

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.sum_weight
        x = self.norm(x)
        x = self.fused_avgpool_gelu.fused_avgpool_gelu_cuda(
            x, 
            self.pool_kernel_size[0], 
            self.pool_kernel_size[1], 
            self.pool_kernel_size[2]
        )
        return x