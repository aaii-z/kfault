import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_avgpool_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_avgpool_sigmoid_sum_kernel(
    const float* conv_output,
    float* output_sum,
    int batch_size,
    int out_channels,
    int conv_H,
    int conv_W,
    int pool_kernel_size
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int H_pooled = conv_H / pool_kernel_size;
    int W_pooled = conv_W / pool_kernel_size;
    int total_elements = out_channels * H_pooled * W_pooled;

    float local_sum = 0.0f;

    for (int i = tid; i < total_elements; i += num_threads) {
        int c = i / (H_pooled * W_pooled);
        int rem = i % (H_pooled * W_pooled);
        int h_p = rem / W_pooled;
        int w_p = rem % W_pooled;

        int h_start = h_p * pool_kernel_size;
        int w_start = w_p * pool_kernel_size;

        float avg = 0.0f;
        for (int kh = 0; kh < pool_kernel_size; ++kh) {
            for (int kw = 0; kw < pool_kernel_size; ++kw) {
                int h_idx = h_start + kh;
                int w_idx = w_start + kw;
                int idx = ((b * out_channels + c) * conv_H + h_idx) * conv_W + w_idx;
                avg += conv_output[idx];
            }
        }
        avg /= (pool_kernel_size * pool_kernel_size);

        float sig = 1.0f / (1.0f + expf(-avg));
        local_sum += sig;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_sum[b] = shared_sum[0];
    }
}

torch::Tensor fused_avgpool_sigmoid_sum_cuda(torch::Tensor conv_output, int pool_kernel_size) {
    auto batch_size = conv_output.size(0);
    auto out_channels = conv_output.size(1);
    auto conv_H = conv_output.size(2);
    auto conv_W = conv_output.size(3);

    auto output_sum = torch::zeros({batch_size}, conv_output.options());

    int block_size = 256;
    int num_blocks = batch_size;

    fused_avgpool_sigmoid_sum_kernel<<<num_blocks, block_size>>>(
        conv_output.data_ptr<float>(),
        output_sum.data_ptr<float>(),
        batch_size,
        out_channels,
        conv_H,
        conv_W,
        pool_kernel_size
    );

    return output_sum;
}
"""

fused_avgpool_sigmoid_sum_cpp = "torch::Tensor fused_avgpool_sigmoid_sum_cuda(torch::Tensor conv_output, int pool_kernel_size);"

fused_avgpool_sigmoid_sum = load_inline(
    name="fused_avgpool_sigmoid_sum",
    cpp_sources=fused_avgpool_sigmoid_sum_cpp,
    cuda_sources=fused_avgpool_sigmoid_sum_source,
    functions=["fused_avgpool_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.fused_op = fused_avgpool_sigmoid_sum

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_avgpool_sigmoid_sum_cuda(x, self.pool_kernel_size)
        return x