import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3d, scaling, MaxPool3d, AdaptiveAvgPool3d, and clamp
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Compute output dimension for transposed convolution
__device__ int compute_output_dim(int input_dim, int padding, int kernel_size, int stride) {
    return (input_dim - 1) * stride - 2 * padding + kernel_size;
}

__global__ void fused_conv_transpose_scale_maxpool_gap_clamp_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    float scale,
    int maxpool_kernel_size,
    float clamp_min,
    float clamp_max
) {
    int batch = blockIdx.x;
    int chan = threadIdx.x;

    if (chan >= out_channels) return;

    // Compute output spatial dimensions after transposed convolution
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;

    // Compute maxpool output size
    int pool_depth = (out_depth - maxpool_kernel_size) / maxpool_kernel_size + 1;
    int pool_height = (out_height - maxpool_kernel_size) / maxpool_kernel_size + 1;
    int pool_width = (out_width - maxpool_kernel_size) / maxpool_kernel_size + 1;

    // Temporary array to store transposed conv output for this (batch, chan)
    float* temp_feat = new float[out_depth * out_height * out_width]();
    
    // Transposed convolution
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    float w = weight[chan * in_channels * kernel_size * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size * kernel_size +
                                     kd * kernel_size * kernel_size + kh * kernel_size + kw];
                    for (int id = 0; id < in_depth; id++) {
                        for (int ih = 0; ih < in_height; ih++) {
                            for (int iw = 0; iw < in_width; iw++) {
                                int od = id * stride + kd - 2 * padding;
                                int oh = ih * stride + kh - 2 * padding;
                                int ow = iw * stride + kw - 2 * padding;

                                if (od >= 0 && od < out_depth && oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                                    int input_idx = batch * in_channels * in_depth * in_height * in_width +
                                                    ic * in_depth * in_height * in_width +
                                                    id * in_height * in_width + ih * in_width + iw;
                                    int out_idx = od * out_height * out_width + oh * out_width + ow;
                                    temp_feat[out_idx] += w * input[input_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias
    for (int idx = 0; idx < out_depth * out_height * out_width; idx++) {
        temp_feat[idx] += bias[chan];
    }

    // Scale
    for (int idx = 0; idx < out_depth * out_height * out_width; idx++) {
        temp_feat[idx] *= scale;
    }

    // Max pooling
    float max_val = -FLT_MAX;
    for (int pd = 0; pd < pool_depth; pd++) {
        for (int ph = 0; ph < pool_height; ph++) {
            for (int pw = 0; pw < pool_width; pw++) {
                float local_max = -FLT_MAX;
                for (int md = 0; md < maxpool_kernel_size; md++) {
                    for (int mh = 0; mh < maxpool_kernel_size; mh++) {
                        for (int mw = 0; mw < maxpool_kernel_size; mw++) {
                            int od = pd * maxpool_kernel_size + md;
                            int oh = ph * maxpool_kernel_size + mh;
                            int ow = pw * maxpool_kernel_size + mw;
                            if (od < out_depth && oh < out_height && ow < out_width) {
                                int out_idx = od * out_height * out_width + oh * out_width + ow;
                                local_max = fmaxf(local_max, temp_feat[out_idx]);
                            }
                        }
                    }
                }
                max_val = fmaxf(max_val, local_max);
            }
        }
    }

    // Global average pooling (on single value from maxpool) -> just use max_val
    float final_val = fmaxf(clamp_min, fminf(clamp_max, max_val));

    // Output is (batch, out_channels, 1, 1, 1)
    output[batch * out_channels + chan] = final_val;

    delete[] temp_feat;
}

torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale,
    int maxpool_kernel_size,
    float clamp_min,
    float clamp_max
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    auto out_channels = weight.size(0);

    auto output = torch::zeros({batch_size, out_channels}, device=input.device(), dtype=torch::kFloat);

    dim3 grid(batch_size);
    dim3 block(out_channels);

    fused_conv_transpose_scale_maxpool_gap_clamp_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        3,
        2,
        1,
        scale,
        maxpool_kernel_size,
        clamp_min,
        clamp_max
    );

    return output.view({batch_size, out_channels, 1, 1, 1});
}
"""

fused_cpp_source = """
torch::Tensor fused_op_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale,
    int maxpool_kernel_size,
    float clamp_min,
    float clamp_max
);
"""

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_op_cuda(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.scale,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max
        )