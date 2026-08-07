import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + AvgPool2d + Sigmoid + Sum
fused_conv_avgpool_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_THREADS 1024

__global__ void fused_conv_avgpool_sigmoid_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_height, int in_width,
    int conv_kernel_h, int conv_kernel_w,
    int conv_stride_h, int conv_stride_w,
    int conv_padding_h, int conv_padding_w,
    int out_channels,
    int pool_kernel, int pool_stride,
    int conv_out_h, int conv_out_w,
    int pooled_h, int pooled_w
) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float local_sum = 0.0f;

    int total_pooled_elements = out_channels * pooled_h * pooled_w;

    for (int idx = tid; idx < total_pooled_elements; idx += num_threads) {
        int oc = idx / (pooled_h * pooled_w);
        int rem = idx % (pooled_h * pooled_w);
        int ph = rem / pooled_w;
        int pw = rem % pooled_w;

        int h_start = ph * pool_stride;
        int w_start = pw * pool_stride;

        float pool_sum = 0.0f;
        int pool_count = 0;

        for (int kh = 0; kh < pool_kernel; kh++) {
            int h_pos = h_start + kh;
            if (h_pos < 0 || h_pos >= conv_out_h) continue;
            for (int kw = 0; kw < pool_kernel; kw++) {
                int w_pos = w_start + kw;
                if (w_pos < 0 || w_pos >= conv_out_w) continue;

                float conv_val = 0.0f;
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int ckh = 0; ckh < conv_kernel_h; ckh++) {
                        int ih = h_pos * conv_stride_h + ckh - conv_padding_h;
                        if (ih < 0 || ih >= in_height) continue;
                        for (int ckw = 0; ckw < conv_kernel_w; ckw++) {
                            int iw = w_pos * conv_stride_w + ckw - conv_padding_w;
                            if (iw < 0 || iw >= in_width) continue;

                            int input_idx = batch * in_channels * in_height * in_width +
                                            ic * in_height * in_width +
                                            ih * in_width + iw;

                            int weight_idx = oc * in_channels * conv_kernel_h * conv_kernel_w +
                                             ic * conv_kernel_h * conv_kernel_w +
                                             ckh * conv_kernel_w + ckw;

                            conv_val += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
                conv_val += bias[oc];
                pool_sum += conv_val;
                pool_count++;
            }
        }

        float avg_pooled = (pool_count > 0) ? pool_sum / pool_count : 0.0f;
        float sigmoid_out = 1.0f / (1.0f + expf(-avg_pooled));
        local_sum += sigmoid_out;
    }

    __shared__ float shared_sum[MAX_THREADS];
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch] = shared_sum[0];
    }
}

torch::Tensor fused_conv_avgpool_sigmoid_sum_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_height, int in_width,
    int conv_kernel_h, int conv_kernel_w,
    int conv_stride_h, int conv_stride_w,
    int conv_padding_h, int conv_padding_w,
    int out_channels,
    int pool_kernel, int pool_stride,
    int conv_out_h, int conv_out_w,
    int pooled_h, int pooled_w) {

    auto batch_size = input.size(0);
    auto output = torch::zeros({batch_size}, input.options());

    const int block_size = 256;
    const int num_blocks = batch_size;

    fused_conv_avgpool_sigmoid_sum_kernel<<<num_blocks, block_size>>>(
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
    int pool_kernel, int pool_stride,
    int conv_out_h, int conv_out_w,
    int pooled_h, int pooled_w);
"""

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
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

        self.conv_stride = self.conv.stride
        self.conv_padding = self.conv.padding
        self.pool_kernel = pool_kernel_size
        self.pool_stride = pool_kernel_size

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape

        conv_out_h = (in_height + 2 * self.conv_padding[0] - self.kernel_size) // self.conv_stride[0] + 1
        conv_out_w = (in_width + 2 * self.conv_padding[1] - self.kernel_size) // self.conv_stride[1] + 1
        pooled_h = (conv_out_h - self.pool_kernel) // self.pool_stride + 1
        pooled_w = (conv_out_w - self.pool_kernel) // self.pool_stride + 1

        output = fused_op.fused_conv_avgpool_sigmoid_sum_cuda(
            x,
            self.weight,
            self.bias,
            in_height, in_width,
            self.kernel_size, self.kernel_size,
            self.conv_stride[0], self.conv_stride[1],
            self.conv_padding[0], self.conv_padding[1],
            self.out_channels,
            self.pool_kernel, self.pool_stride,
            conv_out_h, conv_out_w,
            pooled_h, pooled_w
        )

        return output