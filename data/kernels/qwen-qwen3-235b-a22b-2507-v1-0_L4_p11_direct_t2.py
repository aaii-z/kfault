import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose2d + BatchNorm2d + Tanh
fused_conv_bn_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_conv_transpose_bn_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_weight,
    const float* bn_bias,
    const float* bn_running_mean,
    const float* bn_running_var,
    float* output,
    int batch_size, int in_channels, int in_H, int in_W,
    int out_channels, int out_H, int out_W,
    int kernel_size, int stride, int padding,
    int groups,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_H * out_W;
    if (idx >= total_elements) return;

    int n = idx / (out_channels * out_H * out_W);
    int residual = idx % (out_channels * out_H * out_W);
    int c = residual / (out_H * out_W);
    int h = (residual / out_W) % out_H;
    int w = residual % out_W;

    // Map to input coordinates (transposed conv)
    int h_in_center = h * stride - padding;
    int w_in_center = w * stride - padding;

    // ConvTranspose2d computation
    float conv_val = 0.0f;
    int group = c / (out_channels / groups);
    int channels_per_group_in = in_channels / groups;
    int channels_per_group_out = out_channels / groups;
    int in_ch_start = group * channels_per_group_in;
    int out_ch_start = group * channels_per_group_out;

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int h_in = h_in_center + i;
            int w_in = w_in_center + j;
            if (h_in >= 0 && h_in < in_H && w_in >= 0 && w_in < in_W) {
                for (int g_ch = 0; g_ch < channels_per_group_in; ++g_ch) {
                    int in_c = in_ch_start + g_ch;
                    int weight_idx = (c * channels_per_group_in + g_ch) * kernel_size * kernel_size + i * kernel_size + j;
                    conv_val += input[n * in_channels * in_H * in_W + in_c * in_H * in_W + h_in * in_W + w_in] *
                                weight[weight_idx];
                }
            }
        }
    }
    conv_val += bias[c];

    // BatchNorm: y = scale * (x - mean) / sqrt(var + eps) + shift
    float bn_mean = bn_running_mean[c];
    float bn_var = bn_running_var[c];
    float bn_out = (conv_val - bn_mean) / sqrtf(bn_var + eps);
    bn_out = bn_out * bn_weight[c] + bn_bias[c];

    // Tanh activation
    float tanh_out = tanhf(bn_out);

    output[idx] = tanh_out;
}

torch::Tensor fused_conv_bn_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    int64_t in_H, int64_t in_W,
    int64_t out_H, int64_t out_W,
    int64_t kernel_size, int64_t stride, int64_t padding,
    int64_t groups,
    float eps) {
    
    auto batch_size = input.size(0);
    auto total_elements = batch_size * weight.size(0) * out_H * out_W;
    auto output = torch::empty({batch_size, weight.size(0), out_H, out_W}, input.options());

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_transpose_bn_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input.size(1),
        in_H, in_W,
        weight.size(0),
        out_H, out_W,
        kernel_size,
        stride,
        padding,
        groups,
        eps
    );

    return output;
}
"""

fused_conv_bn_tanh_cpp = """
torch::Tensor fused_conv_bn_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    int64_t in_H, int64_t in_W,
    int64_t out_H, int64_t out_W,
    int64_t kernel_size, int64_t stride, int64_t padding,
    int64_t groups,
    float eps);
"""

# Compile the fused kernel
fused_ops = load_inline(
    name="fused_conv_bn_tanh",
    cpp_sources=[fused_conv_bn_tanh_cpp],
    cuda_sources=[fused_conv_bn_tanh_source],
    functions=["fused_conv_bn_tanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        # Initialize the base layers
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        # Register buffers for fused kernel
        self.register_buffer("weight", self.conv_transpose.weight.detach())
        self.register_buffer("bias", self.conv_transpose.bias.detach())
        self.register_buffer("bn_weight", self.batch_norm.weight.detach())
        self.register_buffer("bn_bias", self.batch_norm.bias.detach())
        self.register_buffer("bn_running_mean", self.batch_norm.running_mean.detach())
        self.register_buffer("bn_running_var", self.batch_norm.running_var.detach())
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.eps = 1e-5
        
        self.fused_op = fused_ops.fused_conv_bn_tanh_cuda

    def forward(self, x):
        # Compute output spatial dimensions for transposed conv
        out_H = (x.size(2) - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_W = (x.size(3) - 1) * self.stride - 2 * self.padding + self.kernel_size
        
        # Fused ConvTranspose + BatchNorm + Tanh
        y = self.fused_op(
            x, self.weight, self.bias,
            self.bn_weight, self.bn_bias,
            self.bn_running_mean, self.bn_running_var,
            x.size(2), x.size(3),
            out_H, out_W,
            self.kernel_size, self.stride, self.padding,
            self.groups,
            self.eps
        )
        
        # MaxPool2d
        y = self.max_pool(y)
        
        # GroupNorm
        y = self.group_norm(y)
        
        return y