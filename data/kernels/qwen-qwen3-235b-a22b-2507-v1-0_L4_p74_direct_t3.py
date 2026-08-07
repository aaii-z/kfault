import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d + LeakyReLU + Multiply + LeakyReLU + MaxPool3d
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

extern "C" __global__ void fused_convtrans_leakyrelu_mul_leakyrelu_maxpool(
    const float* input,
    const float* weight,
    const float* bias,
    const float* multiplier,
    float* output,
    // Input dimensions
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions (after conv transpose)
    int out_depth, int out_height, int out_width,
    // Conv transpose params
    int kernel_size, int stride, int padding, int output_padding,
    // Maxpool params
    int pool_kernel, int pool_stride,
    // Pooled output dimensions
    int pooled_depth, int pooled_height, int pooled_width,
    int out_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    
    if (idx >= total_elements) return;

    int wo = idx;
    int w_pooled = wo % pooled_width; wo /= pooled_width;
    int h_pooled = wo % pooled_height; wo /= pooled_height;
    int d_pooled = wo % pooled_depth; wo /= pooled_depth;
    int c = wo % out_channels;
    int b = wo / out_channels;

    // Map pooled indices to pre-pooling activation indices
    int d_start = d_pooled * pool_stride;
    int h_start = h_pooled * pool_stride;
    int w_start = w_pooled * pool_stride;
    int d_end = d_start + pool_kernel;
    int h_end = h_start + pool_kernel;
    int w_end = w_start + pool_kernel;

    float max_val = -FLT_MAX;

    for (int dk = d_start; dk < d_end; ++dk) {
        for (int hk = h_start; hk < h_end; ++hk) {
            for (int wk = w_start; wk < w_end; ++wk) {
                if (dk < 0 || dk >= out_depth || hk < 0 || hk >= out_height || wk < 0 || wk >= out_width) continue;

                float conv_val = 0.0f;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kz = 0; kz < kernel_size; ++kz) {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int di = dk - kz + padding;
                                int hi = hk - ky + padding;
                                int wi = wk - kx + padding;

                                if (di < 0 || di >= in_depth || hi < 0 || hi >= in_height || wi < 0 || wi >= in_width) continue;

                                if ((di % stride) != 0 || (hi % stride) != 0 || (wi % stride) != 0) continue;

                                di /= stride; hi /= stride; wi /= stride;

                                int in_idx = b * in_channels * in_depth * in_height * in_width +
                                           ic * in_depth * in_height * in_width +
                                           di * in_height * in_width +
                                           hi * in_width +
                                           wi;

                                int weight_idx = c * in_channels * kernel_size * kernel_size * kernel_size +
                                               ic * kernel_size * kernel_size * kernel_size +
                                               kz * kernel_size * kernel_size +
                                               ky * kernel_size +
                                               kx;

                                conv_val += input[in_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                conv_val += bias[c];

                float lr1 = (conv_val > 0) ? conv_val : 0.2f * conv_val;
                float mul_val = lr1 * multiplier[c];
                float lr2 = (mul_val > 0) ? mul_val : 0.2f * mul_val;

                max_val = MAX(max_val, lr2);
            }
        }
    }

    int out_idx = b * out_channels * pooled_depth * pooled_height * pooled_width +
                  c * pooled_depth * pooled_height * pooled_width +
                  d_pooled * pooled_height * pooled_width +
                  h_pooled * pooled_width +
                  w_pooled;
    output[out_idx] = max_val;
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    int in_depth, int in_height, int in_width,
    int kernel_size, int stride, int padding, int output_padding,
    int pool_kernel, int pool_stride
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);

    // Compute output dimensions of conv transpose
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Compute pooled output dimensions
    int pooled_depth = (out_depth - pool_kernel) / pool_stride + 1;
    int pooled_height = (out_height - pool_kernel) / pool_stride + 1;
    int pooled_width = (out_width - pool_kernel) / pool_stride + 1;

    auto output = torch::zeros({batch_size, out_channels, pooled_depth, pooled_height, pooled_width}, input.options());

    const int total_elements = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    fused_convtrans_leakyrelu_mul_leakyrelu_maxpool<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding, output_padding,
        pool_kernel, pool_stride,
        pooled_depth, pooled_height, pooled_width,
        out_channels
    );

    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    int in_depth, int in_height, int in_width,
    int kernel_size, int stride, int padding, int output_padding,
    int pool_kernel, int pool_stride
);
"""

fused_op = load_inline(
    name="fused_op",
    cpp_sources=[fused_cpp_source],
    cuda_sources=[fused_kernel_source],
    functions=["fused_op_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        self.pool_kernel = 2
        self.pool_stride = 2

        self.fused_op = fused_op.fused_op_cuda

    def forward(self, x):
        in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]

        output = self.fused_op(
            x,
            self.weight,
            self.bias,
            self.multiplier,
            in_depth, in_height, in_width,
            self.kernel_size, self.stride, self.padding, self.output_padding,
            self.pool_kernel, self.pool_stride
        )
        return output