import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Conv2d + GroupNorm + Scale + MaxPool2d + Clamp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_conv_gn_scale_pool_clamp_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* scale,
    const float* group_mean,
    const float* group_var,
    float* output,
    // Input tensor shape
    int batch_size, int in_channels, int in_height, int in_width,
    // Output tensor shape (after pooling)
    int out_batch, int out_channels, int out_height, int out_width,
    // Conv parameters
    int kernel_size_h, int kernel_size_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    // Group norm params
    int num_groups,
    // Pooling params
    int pool_kernel_h, int pool_kernel_w,
    int pool_stride_h, int pool_stride_w,
    // Clamp bounds
    float clamp_min, float clamp_max
) {
    int total_elements = out_batch * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decode output index (after pooling)
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);

    // Compute conv output spatial dimensions
    int conv_out_h = (in_height + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / stride_h + 1;
    int conv_out_w = (in_width + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / stride_w + 1;

    // Pooling window start in conv output space
    int conv_h_start = h * pool_stride_h;
    int conv_w_start = w * pool_stride_w;

    // Group index for GroupNorm
    int group_id = c / (out_channels / num_groups);
    float mean = group_mean[group_id];
    float inv_std = 1.0f / sqrtf(group_var[group_id] + 1e-5f);

    float max_val = -FLT_MAX;

    // Max pooling over the pooling window
    for (int ph = 0; ph < pool_kernel_h; ++ph) {
        for (int pw = 0; pw < pool_kernel_w; ++pw) {
            int ch = conv_h_start + ph;
            int cw = conv_w_start + pw;
            if (ch < 0 || ch >= conv_out_h || cw < 0 || cw >= conv_out_w) continue;

            // Perform convolution at (ch, cw)
            float conv_val = 0.0f;
            for (int kh = 0; kh < kernel_size_h; ++kh) {
                for (int kw = 0; kw < kernel_size_w; ++kw) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int ih = ch * stride_h - padding_h + kh * dilation_h;
                        int iw = cw * stride_w - padding_w + kw * dilation_w;
                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            int in_idx = n * in_channels * in_height * in_width +
                                        ic * in_height * in_width +
                                        ih * in_width + iw;
                            int w_idx = c * in_channels * kernel_size_h * kernel_size_w +
                                       ic * kernel_size_h * kernel_size_w +
                                       kh * kernel_size_w + kw;
                            conv_val += input[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
            conv_val += bias[c];

            // Apply GroupNorm and Scale
            float normalized = (conv_val - mean) * inv_std;
            float scaled = normalized * scale[c];

            max_val = fmaxf(max_val, scaled);
        }
    }

    // Clamp
    max_val = fmaxf(clamp_min, fminf(clamp_max, max_val));
    output[idx] = max_val;
}

torch::Tensor fused_conv_gn_scale_pool_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor group_mean,
    torch::Tensor group_var,
    int64_t batch_size, int64_t in_channels, int64_t in_height, int64_t in_width,
    int64_t out_channels, int64_t out_height, int64_t out_width,
    int64_t kernel_size_h, int64_t kernel_size_w,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t num_groups,
    int64_t pool_kernel_h, int64_t pool_kernel_w,
    int64_t pool_stride_h, int64_t pool_stride_w,
    float clamp_min, float clamp_max) {
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_conv_gn_scale_pool_clamp_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        group_mean.data_ptr<float>(),
        group_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        batch_size, out_channels, out_height, out_width,
        kernel_size_h, kernel_size_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        num_groups,
        pool_kernel_h, pool_kernel_w,
        pool_stride_h, pool_stride_w,
        clamp_min, clamp_max
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_conv_gn_scale_pool_clamp_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor group_mean,
    torch::Tensor group_var,
    int64_t batch_size, int64_t in_channels, int64_t in_height, int64_t in_width,
    int64_t out_channels, int64_t out_height, int64_t out_width,
    int64_t kernel_size_h, int64_t kernel_size_w,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t num_groups,
    int64_t pool_kernel_h, int64_t pool_kernel_w,
    int64_t pool_stride_h, int64_t pool_stride_w,
    float clamp_min, float clamp_max);
"""

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

        # Conv2d layer (we'll use its params)
        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

        # Buffers for group statistics
        self.register_buffer('group_mean', torch.zeros(num_groups))
        self.register_buffer('group_var', torch.ones(num_groups))

    def forward(self, x):
        batch_size, _, in_height, in_width = x.shape

        # Conv output size
        conv_out_h = (in_height + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1
        conv_out_w = (in_width + 2 * self.conv.padding[1] - self.conv.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.conv.stride[1] + 1

        # Pooled output size
        out_height = (conv_out_h - self.maxpool_kernel_size[0]) // self.maxpool_kernel_size[0] + 1
        out_width = (conv_out_w - self.maxpool_kernel_size[1]) // self.maxpool_kernel_size[1] + 1

        # Update group statistics using current batch (inference mode)
        with torch.no_grad():
            x_conv = self.conv(x)
            x_conv_grouped = x_conv.view(batch_size, self.num_groups, -1, x_conv.size(2), x_conv.size(3))
            self.group_mean.copy_(x_conv_grouped.mean(dim=[0, 2, 3, 4]))
            self.group_var.copy_(x_conv_grouped.var(dim=[0, 2, 3, 4], unbiased=False))

        # Call fused kernel
        output = fused_op.fused_conv_gn_scale_pool_clamp_cuda(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scale,
            self.group_mean,
            self.group_var,
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