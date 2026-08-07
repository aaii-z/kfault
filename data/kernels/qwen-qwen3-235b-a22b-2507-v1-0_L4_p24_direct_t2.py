import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 3D convolution, min reduction, and softmax
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void check_errors(const char* msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s: %s\\n", msg, cudaGetErrorString(err));
  }
}

__global__ void conv3d_min_softmax_kernel(
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
    int kernel_size,
    int pad
) {
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    if (spatial_idx >= out_height * out_width) return;

    int h = spatial_idx / out_width;
    int w = spatial_idx % out_width;

    float* local_conv_out = new float[out_depth];
    for (int d = 0; d < out_depth; ++d) {
        local_conv_out[d] = bias[channel_idx];
    }

    int k_half = kernel_size / 2;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int weight_idx = channel_idx * (in_channels * kernel_size * kernel_size * kernel_size) +
                                     ic * (kernel_size * kernel_size * kernel_size) +
                                     kd * (kernel_size * kernel_size) +
                                     kh * kernel_size + kw;

                    float w_val = weight[weight_idx];
                    for (int d = 0; d < out_depth; ++d) {
                        int id = d - k_half + kd;
                        int ih = h - k_half + kh;
                        int iw = w - k_half + kw;

                        if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            int input_idx = batch_idx * (in_channels * in_depth * in_height * in_width) +
                                          ic * (in_depth * in_height * in_width) +
                                          id * (in_height * in_width) +
                                          ih * in_width + iw;
                            local_conv_out[d] += w_val * input[input_idx];
                        }
                    }
                }
            }
        }
    }

    float min_val = FLT_MAX;
    for (int d = 0; d < out_depth; ++d) {
        min_val = fminf(min_val, local_conv_out[d]);
    }

    float max_val = -FLT_MAX;
    for (int c = 0; c < out_channels; ++c) {
        int temp_idx = batch_idx * (out_channels * out_height * out_width) +
                       c * (out_height * out_width) +
                       h * out_width + w;
        float val = (c == channel_idx) ? min_val : 0.0f;
        if (val > max_val) max_val = val;
    }

    float sum_val = 0.0f;
    float exp_val = expf(min_val - max_val);
    sum_val += exp_val;

    __shared__ float shared_sum[32];
    __shared__ float shared_max[32];
    int tid = threadIdx.x;
    if (tid == 0) {
        shared_sum[0] = sum_val;
        shared_max[0] = max_val;
    }
    __syncthreads();

    if (tid == 0) {
        sum_val = shared_sum[0];
        max_val = shared_max[0];
    }
    __syncthreads();

    for (int c = 0; c < out_channels; ++c) {
        int out_idx = batch_idx * (out_channels * out_height * out_width) +
                      c * (out_height * out_width) +
                      h * out_width + w;
        float val = (c == channel_idx) ? min_val : 0.0f;
        output[out_idx] = expf(val - max_val) / sum_val;
    }

    delete[] local_conv_out;
}

std::vector<torch::Tensor> fused_conv_min_softmax_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels,
    int out_channels,
    int kernel_size,
    int pad,
    int batch_size,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width) {

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

    dim3 block(256);
    dim3 grid((out_height * out_width + 255) / 256, out_channels, batch_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    conv3d_min_softmax_kernel<<<grid, block, 0, stream>>>(
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
        kernel_size,
        pad
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
    int in_channels,
    int out_channels,
    int kernel_size,
    int pad,
    int batch_size,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width);
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
            self.in_channels, self.out_channels, self.kernel_size, self.pad,
            batch_size, in_depth, in_height, in_width,
            out_depth, out_height, out_width
        )
        return result[0]