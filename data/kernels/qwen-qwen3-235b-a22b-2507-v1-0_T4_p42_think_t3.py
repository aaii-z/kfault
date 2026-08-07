import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

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
    int total = batch * in_channels;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    if (idx >= total) return;

    int b = idx / in_channels;
    int m = idx % in_channels;

    float sum = 0.0f;
    int spatial_size = H * W;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int x_idx = b * in_channels * H * W + m * H * W + i;
        sum += x[x_idx];
    }

    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[b * in_channels + m] = sdata[0];
    }
}

__global__ void compute_gap_output_kernel(
    const float* input_sum,
    const float* K_sum,
    const float* bias_conv,
    float* output,
    int batch,
    int out_channels,
    int in_channels,
    float total_pixels) 
{
    int total = batch * out_channels;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    if (idx >= total) return;

    int b = idx / out_channels;
    int c = idx % out_channels;

    float sum_val = 0.0f;
    for (int m = 0; m < in_channels; m++) {
        sum_val += input_sum[b * in_channels + m] * K_sum[c * in_channels + m];
    }
    sum_val += total_pixels * bias_conv[c];
    output[idx] = sum_val / total_pixels;
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
    float total_pixels = static_cast<float>(H_out * W_out);
    
    int block_size1 = 256;
    int num_blocks1 = (batch * in_channels + block_size1 - 1) / block_size1;
    reduce_spatial_sum_kernel<<<num_blocks1, block_size1>>>(
        x.data_ptr<float>(),
        input_sum.data_ptr<float>(),
        batch, in_channels, H, W
    );
    
    auto output_flat = torch::zeros({batch * out_channels}, x.options());
    int block_size2 = 256;
    int num_blocks2 = (batch * out_channels + block_size2 - 1) / block_size2;
    compute_gap_output_kernel<<<num_blocks2, block_size2>>>(
        input_sum.data_ptr<float>(),
        K_sum.data_ptr<float>(),
        bias_conv.data_ptr<float>(),
        output_flat.data_ptr<float>(),
        batch, out_channels, in_channels, total_pixels
    );
    
    return output_flat.view({batch, out_channels, 1, 1});
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
    if (b >= batch) return;

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
    
    output[b] = 10.0f * (max_val + logf(sum_exp));
}

torch::Tensor fused_bias_logsumexp_mul_cuda(
    torch::Tensor x, 
    torch::Tensor bias) 
{
    int batch = x.size(0);
    int out_channels = x.size(1);
    
    auto output = torch::zeros({batch}, x.options());
    
    int block_size = 256;
    int num_blocks = (batch + block_size - 1) / block_size;
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
        return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.fused_conv = FusedConvTransposeGap(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.fused_conv(x)
        x = x + self.bias
        x = fused_bias_logsumexp_mul.fused_bias_logsumexp_mul_cuda(x.squeeze(3).squeeze(2), self.bias.view(-1))
        return x.unsqueeze(1)