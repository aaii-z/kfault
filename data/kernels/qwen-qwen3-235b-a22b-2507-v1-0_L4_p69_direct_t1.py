import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + Hardswish + ReLU
fused_conv_hardswish_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Fused kernel: Conv2d (with weight and bias) + Hardswish + ReLU
__global__ void fused_conv_hardswish_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size_h, int kernel_size_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c_out = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    int k_h_half = kernel_size_h / 2;
    int k_w_half = kernel_size_w / 2;

    float val = 0.0f;

    // Perform convolution
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size_h; kh++) {
            for (int kw = 0; kw < kernel_size_w; kw++) {
                int ih = h + kh - k_h_half;
                int iw = w + kw - k_w_half;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = ((n * in_channels + c_in) * input_height + ih) * input_width + iw;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_size_h + kh) * kernel_size_w + kw;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    val += bias[c_out];

    // Apply Hardswish: x * relu6(x + 3) / 6
    float hardswish_val = val * fmaxf(0.0f, fminf(6.0f, val + 3.0f)) / 6.0f;

    // Then apply ReLU
    output[idx] = fmaxf(0.0f, hardswish_val);
}

torch::Tensor fused_conv_hardswish_relu_cuda(
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
    int kernel_size_h = weight.size(2);
    int kernel_size_w = weight.size(3);

    // Compute output height and width with padding = kernel_size // 2 and stride = 1
    int padding_h = kernel_size_h / 2;
    int padding_w = kernel_size_w / 2;
    int output_height = input_height + 2 * padding_h - kernel_size_h + 1;
    int output_width = input_width + 2 * padding_w - kernel_size_w + 1;

    auto output = torch::zeros(
        {batch_size, out_channels, output_height, output_width},
        input.options()
    );

    int total_elements = batch_size * out_channels * output_height * output_width;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_conv_hardswish_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size_h, kernel_size_w
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}
"""

fused_conv_hardswish_relu_cpp_source = """
torch::Tensor fused_conv_hardswish_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_conv_hardswish_relu",
    cpp_sources=fused_conv_hardswish_relu_cpp_source,
    cuda_sources=fused_conv_hardswish_relu_source,
    functions=["fused_conv_hardswish_relu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + Hardswish + ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return fused_op.fused_conv_hardswish_relu_cuda(x, self.weight, self.bias)