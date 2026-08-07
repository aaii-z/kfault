import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose2d + BatchNorm2d + Tanh + MaxPool2d + GroupNorm
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to calculate 4D index
__device__ int get_index(int n, int c, int h, int w, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

__global__ void fused_conv_bn_tanh_pool_gn_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* bn_weight,
    const float* bn_bias,
    const float* bn_running_mean,
    const float* bn_running_var,
    const float* gn_weight,
    const float* gn_bias,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int groups,
    int num_groups,
    float eps_bn,
    float eps_gn,
    int pool_kernel,
    int pool_stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * (in_height / 2) * (in_width / 2); // after maxpool
    if (idx >= total_elements) return;

    int n = idx / (out_channels * (in_height / 2) * (in_width / 2));
    int residual = idx % (out_channels * (in_height / 2) * (in_width / 2));
    int c = residual / ((in_height / 2) * (in_width / 2));
    int h_pool = (residual / (in_width / 2)) % (in_height / 2);
    int w_pool = residual % (in_width / 2);

    int h_conv = h_pool * 2;
    int w_conv = w_pool * 2;

    // ConvTranspose2d: apply transposed convolution
    float conv_val = 0.0f;
    int group_id = c / (out_channels / groups);
    int channels_per_group = in_channels / groups;
    int start_ch_in = group_id * channels_per_group;
    int start_ch_out = group_id * (out_channels / groups);

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int h_in = h_conv + i - padding;
            int w_in = w_conv + j - padding;
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                for (int g_ch = 0; g_ch < channels_per_group; ++g_ch) {
                    int in_c = start_ch_in + g_ch;
                    int out_c = c - start_ch_out;
                    conv_val += input[n * in_channels * in_height * in_width + in_c * in_height * in_width + h_in * in_width + w_in] *
                                weight[c * channels_per_group * kernel_size * kernel_size + g_ch * kernel_size * kernel_size + i * kernel_size + j];
                }
            }
        }
    }
    conv_val += bias[c];

    // BatchNorm2d: y = (x - mean) / sqrt(var + eps) * weight + bias
    float bn_mean = bn_running_mean[c];
    float bn_var = bn_running_var[c];
    float bn_out = (conv_val - bn_mean) / sqrtf(bn_var + eps_bn);
    bn_out = bn_out * bn_weight[c] + bn_bias[c];

    // Tanh activation
    float tanh_out = tanhf(bn_out);

    // MaxPool2d 2x2, stride 2: we are at pooled location (h_pool, w_pool), so we take the max over 2x2
    // But since we're computing output directly, we simulate that pooling was already applied
    // So we just use tanh_out as the value at this pooled location

    // GroupNorm: normalize over groups of channels
    int group_size = out_channels / num_groups;
    int group_idx = c / group_size;
    int start_c = group_idx * group_size;
    int end_c = start_c + group_size;

    // Compute mean and variance within the group for this spatial location
    float gn_mean = 0.0f, gn_var = 0.0f;
    for (int gc = start_c; gc < end_c; ++gc) {
        float val = (gc == c) ? tanh_out : 
            // We would need all values in the group — but we don't have them here!
            // This is a limitation: GroupNorm requires reduction over channels in the same group.
            // So we cannot fuse it in a pointwise kernel without precomputed stats.
            // Therefore, we assume GroupNorm stats are precomputed (not typical).
            // Instead, we change strategy: do not fuse GroupNorm, or assume it's fused only if stats are available.
            // Given complexity, we will instead compute GroupNorm separately.
            // But for full fusion, we must recompute all values — not feasible in single kernel without storing intermediate.
            // So we break fusion before GroupNorm.
            0.0f;
        gn_mean += val;
    }
    gn_mean /= group_size;
    for (int gc = start_c; gc < end_c; ++gc) {
        float val = (gc == c) ? tanh_out : 0.0f;
        gn_var += (val - gn_mean) * (val - gn_mean);
    }
    gn_var /= group_size;

    float final_val = (tanh_out - gn_mean) / sqrtf(gn_var + eps_gn);
    final_val = final_val * gn_weight[c] + gn_bias[c];

    output[idx] = final_val;
}
"""

# Instead of fusing everything (which is complex due to GroupNorm's reduction), we do:
# Fusion of ConvTranspose2d + BatchNorm2d + Tanh + MaxPool2d
# Then apply GroupNorm separately (PyTorch's GroupNorm is already efficient)

fused_conv_bn_tanh_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_conv_transpose_bn_tanh_maxpool_kernel(
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
    int local_c = c - out_ch_start;

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int h_in = h_in_center + i;
            int w_in = w_in_center + j;
            if (h_in >= 0 && h_in < in_H && w_in >= 0 && w_in < in_W) {
                for (int g_ch = 0; g_ch < channels_per_group_in; ++g_ch) {
                    int in_c = in_ch_start + g_ch;
                    int weight_idx = c * channels_per_group_in * kernel_size * kernel_size +
                                     g_ch * kernel_size * kernel_size + i * kernel_size + j;
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

    // MaxPool2d: 2x2, stride 2 — but we are writing to downsampled output
    // So we need to take max over 2x2 region in the *current* (tanh_out) feature map
    // But we are at output location (h, w) which corresponds to input region [2h:2h+2, 2w:2w+2]
    // However, we are computing one output element at a time — so we must recompute all 4 values?

    // Instead, we change approach: first compute full size after conv+bn+tanh, then do maxpool
    // But that would require two kernels. For full fusion, we need to compute all 4 values here.

    // We are writing to maxpooled output, so we compute the 2x2 block that maps to this output
    // But we are at output (n, c, h, w), so we need to compute inputs at:
    // (n, c, 2h, 2w), (2h,2w+1), (2h+1,2w), (2h+1,2w+1)
    // But we can't do that in one thread without race or recomputation.

    // Alternative: split into two kernels.
    // Given complexity, we instead fuse only ConvTranspose + BN + Tanh, and use PyTorch MaxPool2d.
    // But the problem asks for speedups — so we try to fuse as much as possible.

    // We abandon full fusion and instead do:
    // Kernel 1: ConvTranspose + BN + Tanh (output to intermediate buffer)
    // Kernel 2: MaxPool2d (using PyTorch or custom)
    // Then GroupNorm

    // But we are writing one kernel — so we assume MaxPool is not fused.

    // So we return tanh_out for now, and do maxpool separately.
    // This kernel will only do ConvTranspose + BN + Tanh
    output[idx] = tanh_out;
}
""";

# Revised: Only fuse ConvTranspose2d + BatchNorm2d + Tanh
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
    int local_c = c - out_ch_start;

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int h_in = h_in_center + i;
            int w_in = w_in_center + j;
            if (h_in >= 0 && h_in < in_H && w_in >= 0 && w_in < in_W) {
                for (int g_ch = 0; g_ch < channels_per_group_in; ++g_ch) {
                    int in_c = in_ch_start + g_ch;
                    int weight_idx = c * channels_per_group_in * kernel_size * kernel_size +
                                     g_ch * kernel_size * kernel_size + i * kernel_size + j;
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
""";

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
    float eps
);
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
        # Initialize the parameters from ConvTranspose2d and BatchNorm2d
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        # Extract parameters for CUDA kernel
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
        # Fused ConvTranspose + BatchNorm + Tanh
        out_H = (x.size(2) - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_W = (x.size(3) - 1) * self.stride - 2 * self.padding + self.kernel_size
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
        y = y.view(x.size(0), self.out_channels, out_H, out_W)
        
        # MaxPool2d
        y = self.max_pool(y)
        
        # GroupNorm
        y = self.group_norm(y)
        
        return y