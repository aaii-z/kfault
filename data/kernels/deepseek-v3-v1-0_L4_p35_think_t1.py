import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused convolution + subtract + HardSwish
conv_sub_hardswish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_sub_hardswish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float subtract_value,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride, int padding) {
    
    int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    int total_elements = batch_size * out_channels * output_h * output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    int n = idx / (out_channels * output_h * output_w);
    int c = (idx / (output_h * output_w)) % out_channels;
    int h = (idx / output_w) % output_h;
    int w = idx % output_w;
    
    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = h * stride + kh - padding;
                int iw = w * stride + kw - padding;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = ((n * in_channels + ic) * height + ih) * width + iw;
                    int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[c];
    }
    
    // Subtract value and apply HardSwish
    sum -= subtract_value;
    sum = sum * fminf(fmaxf(0.0f, sum + 3.0f), 6.0f) / 6.0f;
    
    output[idx] = sum;
}

torch::Tensor fused_conv_sub_hardswish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float subtract_value, int stride, int padding) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());
    
    int total_elements = batch_size * out_channels * output_h * output_w;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_sub_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), subtract_value,
        batch_size, in_channels, out_channels,
        height, width, kernel_size, stride, padding);
    
    return output;
}
"""

conv_sub_hardswish_cpp = "torch::Tensor fused_conv_sub_hardswish_cuda(torch::Tensor, torch::Tensor, torch::Tensor, float, int, int);"

# Load the custom CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=conv_sub_hardswish_cpp,
    cuda_sources=conv_sub_hardswish_source,
    functions=['fused_conv_sub_hardswish_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        
        # Initialize weights and bias for convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Set convolution parameters
        self.stride = 1
        self.padding = 0

    def forward(self, x):
        # Use fused convolution + subtract + HardSwish
        x = fused_ops.fused_conv_sub_hardswish_cuda(
            x, self.weight, self.bias,
            self.subtract_value,
            self.stride, self.padding
        )
        
        x = self.pool(x)
        x = torch.nn.functional.mish(x)
        return x