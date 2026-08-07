import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D convolution, min reduction, and softmax
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv3d_min_softmax_kernel(
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
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    if (idx >= out_height * out_width) return;

    int h = idx / out_width;
    int w = idx % out_width;

    float min_val = FLT_MAX;

    // Compute conv3d output for all depth positions and take min
    for (int od = 0; od < out_depth; ++od) {
        float val = bias[channel_idx];
        int k_half = kernel_size / 2;

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int id = od - k_half + kd;
                        int ih = h - k_half + kh;
                        int iw = w - k_half + kw;

                        if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            int input_idx = batch_idx * in_channels * in_depth * in_height * in_width +
                                          ic * in_depth * in_height * in_width +
                                          id * in_height * in_width +
                                          ih * in_width + iw;

                            int weight_idx = channel_idx * in_channels * kernel_size * kernel_size * kernel_size +
                                           ic * kernel_size * kernel_size * kernel_size +
                                           kd * kernel_size * kernel_size +
                                           kh * kernel_size + kw;

                            val += weight[weight_idx] * input[input_idx];
                        }
                    }
                }
            }
        }

        min_val = fminf(min_val, val);
    }

    // Store min value in temporary global memory for this (batch, channel, h, w)
    int temp_idx = batch_idx * out_channels * out_height * out_width +
                   channel_idx * out_height * out_width +
                   h * out_width + w;
    output[temp_idx] = min_val;
}

__global__ void softmax_kernel(
    float* output,
    int batch_size,
    int out_channels,
    int out_height,
    int out_width
) {
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spatial_idx >= out_height * out_width) return;

    int h = spatial_idx / out_width;
    int w = spatial_idx % out_width;
    int batch_idx = blockIdx.y;

    int base_idx = batch_idx * out_channels * out_height * out_width +
                   h * out_width + w;

    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int c = 0; c < out_channels; ++c) {
        float val = output[base_idx + c * out_height * out_width];
        max_val = fmaxf(max_val, val);
    }

    // Compute exp and sum
    float sum_val = 0.0f;
    for (int c = 0; c < out_channels; ++c) {
        float val = output[base_idx + c * out_height * out_width];
        float exp_val = expf(val - max_val);
        output[base_idx + c * out_height * out_width] = exp_val;
        sum_val += exp_val;
    }

    // Normalize
    for (int c = 0; c < out_channels; ++c) {
        output[base_idx + c * out_height * out_width] /= sum_val;
    }
}

std::vector<torch::Tensor> fused_conv_min_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size) {

    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    auto options = torch::TensorOptions()
                      .dtype(input.dtype())
                      .device(input.device())
                      .requires_grad(input.requires_grad());

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, options);

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    dim3 block1(256);
    dim3 grid1((out_height * out_width + 255) / 256, out_channels, batch_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_conv3d_min_softmax_kernel<<<grid1, block1, 0, stream>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        out_depth,
        out_height,
        out_width,
        kernel_size
    );

    dim3 block2(256);
    dim3 grid2((out_height * out_width + 255) / 256, batch_size);

    softmax_kernel<<<grid2, block2, 0, stream>>>(
        output_ptr,
        batch_size,
        out_channels,
        out_height,
        out_width
    );

    return {output};
}
"""

cpp_source = """
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> fused_conv_min_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size);
"""

fused_ops = load_inline(
    name="fused_conv_min_softmax",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_conv_min_softmax_forward"],
    verbose=False,
    with_cuda=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dim = dim
        self.pad = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=self.pad)

    def forward(self, x):
        batch_size, _, in_depth, in_height, in_width = x.shape
        out_depth = in_depth
        out_height = in_height
        out_width = in_width

        result = fused_ops.fused_conv_min_softmax_forward(
            x, self.conv.weight, self.conv.bias,
            batch_size,
            self.in_channels,
            in_depth,
            in_height,
            in_width,
            self.out_channels,
            out_depth,
            out_height,
            out_width,
            self.kernel_size
        )
        return result[0]