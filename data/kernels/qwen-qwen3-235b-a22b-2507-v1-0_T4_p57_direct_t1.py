import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + ReLU + HardSwish
fused_conv_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

__global__ void fused_conv_relu_hardswish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size,
    int padding
) {
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z % batch_size;
    int oc = blockIdx.z / batch_size;

    if (out_h >= output_height || out_w >= output_width || n >= batch_size || oc >= out_channels) {
        return;
    }

    int kh, kw, ic;
    float val = 0.0f;

    // Convolution with padding
    for (kh = 0; kh < kernel_size; ++kh) {
        for (kw = 0; kw < kernel_size; ++kw) {
            for (ic = 0; ic < in_channels; ++ic) {
                int i_h = out_h + padding - kh;
                int i_w = out_w + padding - kw;

                if (i_h >= 0 && i_h < input_height && i_w >= 0 && i_w < input_width) {
                    int in_idx = n * in_channels * input_height * input_width +
                                 ic * input_height * input_width +
                                 i_h * input_width + i_w;
                    int w_idx = oc * in_channels * kernel_size * kernel_size +
                                ic * kernel_size * kernel_size +
                                kh * kernel_size + kw;
                    val += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    // Add bias
    val += bias[oc];

    // Apply ReLU
    val = fmaxf(0.0f, val);

    // Apply HardSwish: x * clamp((x + 3) / 6, 0, 1)
    float hardswish = val * fminf(1.0f, fmaxf(0.0f, (val + 3.0f) / 6.0f));

    // Write output
    int out_idx = n * out_channels * output_height * output_width +
                  oc * output_height * output_width +
                  out_h * output_width + out_w;
    output[out_idx] = hardswish;
}

torch::Tensor fused_conv_relu_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int padding = kernel_size / 2;
    int output_height = input_height;
    int output_width = input_width;

    auto output = torch::empty(
        {batch_size, out_channels, output_height, output_width},
        input.options()
    );

    // Launch configuration
    const int block_size_x = 16;
    const int block_size_y = 16;
    dim3 block_dim(block_size_x, block_size_y);
    dim3 grid_dim(
        (output_width + block_size_x - 1) / block_size_x,
        (output_height + block_size_y - 1) / block_size_y,
        batch_size * out_channels
    );

    fused_conv_relu_hardswish_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size,
        padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}
"""

fused_conv_relu_hardswish_cpp_source = """
torch::Tensor fused_conv_relu_hardswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_relu_hardswish",
    cpp_sources=fused_conv_relu_hardswish_cpp_source,
    cuda_sources=fused_conv_relu_hardswish_source,
    functions=["fused_conv_relu_hardswish_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused Conv2d + ReLU + HardSwish in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Create the weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return fused_op.fused_conv_relu_hardswish_cuda(x, self.weight, self.bias)