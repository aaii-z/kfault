import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

__global__ void fused_conv_activation_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, int in_channels, int height, int width,
    int out_channels, int kernel_size, int pad, int stride,
    int out_height, int out_width
) {
    const int output_size = batch_size * out_channels * out_height * out_width;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= output_size) return;
    
    const int n = tid / (out_channels * out_height * out_width);
    const int c = (tid / (out_height * out_width)) % out_channels;
    const int h = (tid / out_width) % out_height;
    const int w = tid % out_width;
    
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        for (int l = 0; l < kernel_size; ++l) {
            const int h_in = h * stride - pad + k;
            const int w_in = w * stride - pad + l;
            
            if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    const float input_val = input[n * in_channels * height * width + ic * height * width + h_in * width + w_in];
                    const float weight_val = weight[c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + k * kernel_size + l];
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    sum += bias[c];
    
    float x = sum;
    if (x < 0.0f) {
        output[tid] = 0.0f;
    } else {
        if (x <= 3.0f) {
            output[tid] = x * (x + 3.0f) / 6.0f;
        } else {
            output[tid] = x;
        }
    }
}

torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride, int pad, int kernel_size,
    int out_height, int out_width
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());
    
    const int output_size = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int grid_size = DIVUP(output_size, block_size);
    
    fused_conv_activation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, height, width,
        out_channels, kernel_size, pad, stride,
        out_height, out_width
    );
    
    return output;
}
"""

fused_conv_activation_cpp = "torch::Tensor fused_conv_activation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int pad, int kernel_size, int out_height, int out_width);"

fused_conv_activation = load_inline(
    name="fused_conv_activation",
    cpp_sources=fused_conv_activation_cpp,
    cuda_sources=fused_conv_activation_source,
    functions=["fused_conv_activation_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = (kernel_size - 1) // 2
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        return fused_conv_activation.fused_conv_activation_cuda(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.kernel_size,
            out_height, out_width
        )