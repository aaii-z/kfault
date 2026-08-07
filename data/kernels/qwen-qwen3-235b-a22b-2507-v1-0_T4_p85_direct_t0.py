import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + GroupNorm + Scale + MaxPool2d + Clamp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function to calculate 4D index from flattened index
__device__ int get_index(int idx, int dim0, int dim1, int dim2, int dim3) {
    return idx;
}

// Fused kernel: Conv2d (with weight and bias) -> GroupNorm -> Scale -> MaxPool2d -> Clamp
__global__ void fused_conv_gn_scale_pool_clamp(
    const float* input,
    const float* weight,
    const float* bias,
    const float* scale,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size_h, int kernel_size_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int pool_kernel_h, int pool_kernel_w,
    int pool_stride_h, int pool_stride_w,
    float clamp_min, float clamp_max
) {
    int total_elements = batch_size * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);

    // Convolution parameters
    int conv_out_h = (in_height + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / stride_h + 1;
    int conv_out_w = (in_width + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / stride_w + 1;

    // MaxPool output size
    int pooled_h = (conv_out_h - pool_kernel_h) / pool_stride_h + 1;
    int pooled_w = (conv_out_w - pool_kernel_w) / pool_stride_w + 1;

    // Reverse map pooled output to convolution output (before pooling)
    int conv_h_start = h * pool_stride_h;
    int conv_w_start = w * pool_stride_w;
    int conv_h_end = conv_h_start + pool_kernel_h;
    int conv_w_end = conv_w_start + pool_kernel_w;

    float conv_sum = 0.0f;

    // Perform convolution
    for (int kh = 0; kh < kernel_size_h; ++kh) {
        for (int kw = 0; kw < kernel_size_w; ++kw) {
            for (int ic = 0; ic < in_channels; ++ic) {
                int ih = conv_h_start + kh * dilation_h - padding_h;
                int iw = conv_w_start + kw * dilation_w - padding_w;

                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int in_idx = n * in_channels * in_height * in_width +
                                 ic * in_height * in_width +
                                 ih * in_width + iw;
                    int weight_idx = c * in_channels * kernel_size_h * kernel_size_w +
                                     ic * kernel_size_h * kernel_size_w +
                                     kh * kernel_size_w + kw;
                    conv_sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    conv_sum += bias[c];

    // Group Normalization
    int group_id = c / (out_channels / groups);
    int group_size = out_channels / groups;

    // For simplicity, we assume running_mean and running_var are precomputed and stored per group
    float mean = running_mean[group_id];
    float inv_std = 1.0f / sqrt(running_var[group_id] + 1e-5f);
    float gn_output = (conv_sum - mean) * inv_std;

    // Scale
    float scaled_output = gn_output * scale[c];

    // Max pooling over the conv output region
    float max_val = -FLT_MAX;
    for (int ph = 0; ph < pool_kernel_h; ++ph) {
        for (int pw = 0; pw < pool_kernel_w; ++pw) {
            int ch = conv_h_start + ph;
            int cw = conv_w_start + pw;
            if (ch < conv_out_h && cw < conv_out_w) {
                float val = 0.0f;
                for (int kh = 0; kh < kernel_size_h; ++kh) {
                    for (int kw = 0; kw < kernel_size_w; ++kw) {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            int ih = ch + kh * dilation_h - padding_h;
                            int iw = cw + kw * dilation_w - padding_w;
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                int in_idx = n * in_channels * in_height * in_width +
                                             ic * in_height * in_width +
                                             ih * in_width + iw;
                                int weight_idx = c * in_channels * kernel_size_h * kernel_size_w +
                                                 ic * kernel_size_h * kernel_size_w +
                                                 kh * kernel_size_w + kw;
                                val += input[in_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                val += bias[c];
                val = (val - mean) * inv_std * scale[c];  // Apply GN and scale for comparison
                max_val = fmaxf(max_val, val);
            }
        }
    }

    // Clamp
    max_val = fmaxf(clamp_min, fminf(clamp_max, max_val));

    output[idx] = max_val;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_gn_scale_pool_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    int64_t batch_size, int64_t in_channels, int64_t in_height, int64_t in_width,
    int64_t out_channels, int64_t out_height, int64_t out_width,
    int64_t kernel_size_h, int64_t kernel_size_w,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t groups,
    int64_t pool_kernel_h, int64_t pool_kernel_w,
    int64_t pool_stride_h, int64_t pool_stride_w,
    float clamp_min, float clamp_max);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_gn_scale_pool_clamp",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_conv_gn_scale_pool_clamp_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + GroupNorm + Scale + MaxPool2d + Clamp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.num_groups = num_groups
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size if isinstance(maxpool_kernel_size, (list, tuple)) else (maxpool_kernel_size, maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Conv2d parameters
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # GroupNorm
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

        # Precompute running stats (for inference; in training, we'd need to update them)
        self.register_buffer('running_mean', torch.zeros(num_groups))
        self.register_buffer('running_var', torch.ones(num_groups))

    def forward(self, x):
        batch_size, _, in_height, in_width = x.shape

        # Compute output spatial dimensions after convolution
        conv_out_h = (in_height + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1
        conv_out_w = (in_width + 2 * self.conv.padding[1] - self.conv.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.conv.stride[1] + 1

        # After max pooling
        out_height = (conv_out_h - self.maxpool_kernel_size[0]) // self.maxpool_kernel_size[0] + 1
        out_width = (conv_out_w - self.maxpool_kernel_size[1]) // self.maxpool_kernel_size[1] + 1

        # Update running stats for GroupNorm (using current batch stats for inference-style behavior)
        with torch.no_grad():
            x_gn = self.conv(x)
            x_gn = x_gn.view(batch_size, self.num_groups, -1)
            self.running_mean.copy_(x_gn.mean(dim=2).mean(0))
            self.running_var.copy_(x_gn.var(dim=2, unbiased=False).mean(0))

        # Call fused CUDA kernel
        output = fused_op.fused_conv_gn_scale_pool_clamp_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scale,
            self.running_mean,
            self.running_var,
            batch_size, self.in_channels, in_height, in_width,
            self.out_channels, out_height, out_width,
            self.kernel_size[0], self.kernel_size[1],
            self.conv.stride[0], self.conv.stride[1],
            self.conv.padding[0], self.conv.padding[1],
            self.conv.dilation[0], self.conv.dilation[1],
            self.num_groups,
            self.maxpool_kernel_size[0], self.maxpool_kernel_size[1],
            self.maxpool_kernel_size[0], self.maxpool_kernel_size[1],
            self.clamp_min, self.clamp_max
        )

        return output