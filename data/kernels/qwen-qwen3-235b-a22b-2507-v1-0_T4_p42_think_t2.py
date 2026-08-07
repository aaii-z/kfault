import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused_conv_transpose_gap
fused_conv_transpose_gap_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void reduce_spatial_sum_kernel(
    const float* x, 
    float* y, 
    int batch, 
    int in_channels, 
    int H, 
    int W) 
{
    int b = blockIdx.y;
    int m = blockIdx.x;
    if (b < batch && m < in_channels) {
        float sum = 0.0f;
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                int idx = b * in_channels * H * W + m * H * W + i * W + j;
                sum += x[idx];
            }
        }
        y[b * in_channels + m] = sum;
    }
}

__global__ void fused_conv_transpose_gap_step2_kernel(
    const float* input_sum,
    const float* bias_conv,
    const float* K_sum,
    float* output,
    int batch,
    int out_channels,
    int in_channels,
    float total_pixels) 
{
    int b = blockIdx.y;
    int c = blockIdx.x;
    if (b < batch && c < out_channels) {
        float sum_val = 0.0f;
        for (int m = 0; m < in_channels; m++) {
            sum_val += input_sum[b * in_channels + m] * K_sum[c * in_channels + m];
        }
        sum_val += total_pixels * bias_conv[c];
        output[b * out_channels + c] = sum_val / total_pixels;
    }
}

torch::Tensor fused_conv_transpose_gap_cuda(
    torch::Tensor x, 
    torch::Tensor bias_conv, 
    torch::Tensor K_sum,
    int H_out, 
    int W_out) 
{
    int batch = x.size(0);
    int in_channels = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int out_channels = bias_conv.size(0);
    
    auto input_sum = torch::zeros({batch, in_channels}, x.options());
    auto output = torch::zeros({batch, out_channels}, x.options());
    
    const int block_size_x = 32;
    const int block_size_y = 32;
    dim3 block_dim(
        (in_channels + block_size_x - 1) / block_size_x,
        (batch + block_size_y - 1) / block_size_y
    );
    dim3 grid_dim(
        (out_channels + block_size_x - 1) / block_size_x,
        (batch + block_size_y - 1) / block_size_y
    );
    
    float total_pixels = static_cast<float>(H_out * W_out);
    
    reduce_spatial_sum_kernel<<<block_dim, dim3(block_size_x, block_size_y)>>>(
        x.data_ptr<float>(),
        input_sum.data_ptr<float>(),
        batch, in_channels, H, W
    );
    
    fused_conv_transpose_gap_step2_kernel<<<grid_dim, dim3(block_size_x, block_size_y)>>>(
        input_sum.data_ptr<float>(),
        bias_conv.data_ptr<float>(),
        K_sum.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, out_channels, in_channels, total_pixels
    );
    
    return output;
}
"""

fused_conv_transpose_gap_cpp = """
torch::Tensor fused_conv_transpose_gap_cuda(
    torch::Tensor x, 
    torch::Tensor bias_conv, 
    torch::Tensor K_sum,
    int H_out, 
    int W_out);
"""

# Custom CUDA kernel for fused_bias_logsumexp_mul
fused_bias_logsumexp_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bias_logsumexp_mul_kernel(
    const float* x,
    const float* bias,
    float* output,
    int batch,
    int out_channels) 
{
    int b = blockIdx.x;
    if (b < batch) {
        float max_val = -INFINITY;
        for (int c = 0; c < out_channels; c++) {
            float val = x[b * out_channels + c] + bias[c];
            if (val > max_val) max_val = val;
        }
        
        float sum_exp = 0.0f;
        for (int c = 0; c < out_channels; c++) {
            float val = x[b * out_channels + c] + bias[c];
            sum_exp += expf(val - max_val);
        }
        
        output[b] = (max_val + logf(sum_exp)) * 10.0f;
    }
}

torch::Tensor fused_bias_logsumexp_mul_cuda(
    torch::Tensor x, 
    torch::Tensor bias) 
{
    int batch = x.size(0);
    int out_channels = x.size(1);
    
    auto output = torch::zeros({batch}, x.options());
    
    const int block_size = 256;
    const int num_blocks = (batch + block_size - 1) / block_size;
    
    fused_bias_logsumexp_mul_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, out_channels
    );
    
    return output;
}
"""

fused_bias_logsumexp_mul_cpp = """
torch::Tensor fused_bias_logsumexp_mul_cuda(
    torch::Tensor x, 
    torch::Tensor bias);
"""

# Compile custom CUDA operators
fused_conv_transpose_gap = load_inline(
    name="fused_conv_transpose_gap",
    cpp_sources=fused_conv_transpose_gap_cpp,
    cuda_sources=fused_conv_transpose_gap_source,
    functions=["fused_conv_transpose_gap_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)

fused_bias_logsumexp_mul = load_inline(
    name="fused_bias_logsumexp_mul",
    cpp_sources=fused_bias_logsumexp_mul_cpp,
    cuda_sources=fused_bias_logsumexp_mul_source,
    functions=["fused_bias_logsumexp_mul_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)

class FusedConvTransposeGap(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        with torch.no_grad():
            weight = self.conv.weight
            self.K_sum = weight.sum(dim=[2, 3]).contiguous()
            self.register_buffer('bias_conv', self.conv.bias)
    
    def forward(self, x):
        batch, _, H, W = x.shape
        H_out = (H - 1) * self.conv.stride[0] - 2 * self.conv.padding[0] + self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) + self.conv.output_padding[0] + 1
        W_out = (W - 1) * self.conv.stride[1] - 2 * self.conv.padding[1] + self.conv.dilation[1] * (self.conv.kernel_size[1] - 1) + self.conv.output_padding[1] + 1
        output = fused_conv_transpose_gap.fused_conv_transpose_gap_cuda(
            x, self.bias_conv, self.K_sum, H_out, W_out
        )
        return output.view(batch, -1, 1, 1)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.fused_conv = FusedConvTransposeGap(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.fused_conv(x)
        x = fused_bias_logsumexp_mul.fused_bias_logsumexp_mul_cuda(x.squeeze(3).squeeze(2), self.bias.view(-1))
        return x.unsqueeze(1)