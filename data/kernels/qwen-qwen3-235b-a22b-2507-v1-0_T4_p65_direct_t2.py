import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + AvgPool2d + Sigmoid + Sum with high parallelism
fused_conv_avgpool_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_THREADS 1024

// Efficient device function for sigmoid
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-fmaxf(-88.0f, fminf(88.0f, x))));
}

// Main fused kernel with per-output-element parallelism
__global__ void fused_conv_avgpool_sigmoid_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    // Input dimensions
    int batch_size, int in_channels, int in_height, int in_width,
    // Conv parameters
    int conv_kernel_h, int conv_kernel_w,
    int conv_stride_h, int conv_stride_w,
    int conv_padding_h, int conv_padding_w,
    int out_channels,
    // Pooling parameters
    int pool_kernel, int pool_stride,
    // Precomputed sizes
    int conv_out_h, int conv_out_w,
    int pooled_h, int pooled_w
) {
    int batch = blockIdx.x;
    int elem = blockIdx.y * blockDim.x + threadIdx.x;
    int total_elems = out_channels * pooled_h * pooled_w;
    
    if (batch >= batch_size || elem >= total_elems) return;

    int oc = elem / (pooled_h * pooled_w);
    int ph = (elem % (pooled_h * pooled_w)) / pooled_w;
    int pw = elem % pooled_w;

    float conv_value = 0.0f;

    // Convolution computation at all spatial positions covered by pooling kernel
    int h_start = ph * pool_stride;
    int w_start = pw * pool_stride;

    float pool_sum = 0.0f;
    int pool_count = 0;

    for (int kh = 0; kh < pool_kernel; ++kh) {
        for (int kw = 0; kw < pool_kernel; ++kw) {
            int h_pos = h_start + kh;
            int w_pos = w_start + kw;

            if (h_pos < 0 || h_pos >= conv_out_h || w_pos < 0 || w_pos >= conv_out_w) continue;

            float val = 0.0f;
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int ckh = 0; ckh < conv_kernel_h; ++ckh) {
                    for (int ckw = 0; ckw < conv_kernel_w; ++ckw) {
                        int ih = h_pos * conv_stride_h + ckh - conv_padding_h;
                        int iw = w_pos * conv_stride_w + ckw - conv_padding_w;

                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            int input_idx = ((batch * in_channels + ic) * in_height + ih) * in_width + iw;
                            int weight_idx = ((oc * in_channels + ic) * conv_kernel_h + ckh) * conv_kernel_w + ckw;
                            val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            val += bias[oc];
            pool_sum += val;
            ++pool_count;
        }
    }

    float avg_pooled = pool_count > 0 ? pool_sum / (pool_kernel * pool_kernel) : 0.0f;
    float result = sigmoid(avg_pooled);

    // Use atomics to accumulate sum across all output elements for this batch
    atomicAdd(&output[batch], result);
}

// C++ wrapper
torch::Tensor fused_conv_avgpool_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_height, int in_width,
    int conv_kernel_h, int conv_kernel_w,
    int conv_stride_h, int conv_stride_w,
    int conv_padding_h, int conv_padding_w,
    int out_channels,
    int pool_kernel, int pool_stride
) {
    auto batch_size = input.size(0);
    auto options = input.options();
    auto output = torch::zeros({batch_size}, options);

    // Precompute output dimensions
    int conv_out_h = (in_height + 2 * conv_padding_h - conv_kernel_h) / conv_stride_h + 1;
    int conv_out_w = (in_width + 2 * conv_padding_w - conv_kernel_w) / conv_stride_w + 1;
    int pooled_h = (conv_out_h - pool_kernel) / pool_stride + 1;
    int pooled_w = (conv_out_w - pool_kernel) / pool_stride + 1;

    int total_elems = out_channels * pooled_h * pooled_w;
    int block_size = min(max(256, (total_elems + 31) / 32), MAX_THREADS);
    dim3 block_dim(block_size);
    dim3 grid_dim(batch_size, (total_elems + block_size - 1) / block_size);

    fused_conv_avgpool_sigmoid_sum_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, input.size(1), in_height, in_width,
        conv_kernel_h, conv_kernel_w,
        conv_stride_h, conv_stride_w,
        conv_padding_h, conv_padding_w,
        out_channels,
        pool_kernel, pool_stride,
        conv_out_h, conv_out_w,
        pooled_h, pooled_w
    );

    return output;
}
"""

fused_conv_avgpool_sigmoid_sum_cpp = """
torch::Tensor fused_conv_avgpool_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_height, int in_width,
    int conv_kernel_h, int conv_kernel_w,
    int conv_stride_h, int conv_stride_w,
    int conv_padding_h, int conv_padding_w,
    int out_channels,
    int pool_kernel, int pool_stride);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_avgpool_sigmoid_sum",
    cpp_sources=fused_conv_avgpool_sigmoid_sum_cpp,
    cuda_sources=fused_conv_avgpool_sigmoid_sum_source,
    functions=["fused_conv_avgpool_sigmoid_sum_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + AvgPool2d + Sigmoid + Sum.
    Uses high parallelism: one thread per output element (oc, ph, pw).
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

        # Create conv layer to reuse parameters
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

        self.conv_stride = self.conv.stride
        self.conv_padding = self.conv.padding
        self.pool_kernel = pool_kernel_size
        self.pool_stride = pool_kernel_size

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape

        output = fused_op.fused_conv_avgpool_sigmoid_sum_cuda(
            x,
            self.weight,
            self.bias,
            in_height, in_width,
            self.kernel_size, self.kernel_size,
            self.conv_stride[0], self.conv_stride[1],
            self.conv_padding[0], self.conv_padding[1],
            self.out_channels,
            self.pool_kernel, self.pool_stride
        )
        return output