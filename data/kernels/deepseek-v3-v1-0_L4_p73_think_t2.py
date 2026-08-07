import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_bn_scale_kernel(
    const float* input,
    const float* weight,
    const float* scale,
    const float* bias,
    float* output,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int height,
    const int width,
    const int output_height,
    const int output_width) {
    
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int w_out = blockIdx.z * blockDim.z + threadIdx.z;
    const int c_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c_out >= out_channels || h_out >= output_height || w_out >= output_width) return;

    const int input_size = height * width;
    const int kernel_elements = kernel_size * kernel_size;
    const int output_size = output_height * output_width;

    float value = 0.0f;
    const int pad = kernel_size / 2;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int h_in = h_out + kh - pad;
                const int w_in = w_out + kw - pad;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    const int input_idx = c_in * input_size + h_in * width + w_in;
                    const int weight_idx = c_out * (in_channels * kernel_elements) + 
                                       c_in * kernel_elements + 
                                       kh * kernel_size + kw;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    const int output_idx = c_out * output_size + h_out * output_width + w_out;
    output[output_idx] = value * scale[c_out] + bias[c_out];
}

torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor bias,
    int kernel_size) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(scale);
    CHECK_INPUT(bias);

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int output_height = height;
    const int output_width = width;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    const dim3 blocks(
        (out_channels + 7) / 8,
        (output_height + 7) / 8,
        (output_width + 7) / 8
    );
    const dim3 threads(8, 8, 8);

    for (int b = 0; b < batch_size; ++b) {
        fused_conv_bn_scale_kernel<<<blocks, threads>>>(
            input[b].data_ptr<float>(),
            weight.data_ptr<float>(),
            scale.data_ptr<float>(),
            bias.data_ptr<float>(),
            output[b].data_ptr<float>(),
            in_channels,
            out_channels,
            kernel_size,
            height,
            width,
            output_height,
            output_width
        );
    }
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor bias,
    int kernel_size);
"""

fused_conv_bn_scale = load_inline(
    name='fused_conv_bn_scale',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_conv_bn_scale_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        self.eps = 1e-5

        self.conv_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.Tensor(out_channels))
        self.bn_weight = nn.Parameter(torch.Tensor(out_channels))
        self.bn_bias = nn.Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv_bias, -bound, bound)
        nn.init.ones_(self.bn_weight)
        nn.init.zeros_(self.bn_bias)

    def forward(self, x):
        scale = (self.scaling_factor * self.bn_weight) / torch.sqrt(self.running_var + self.eps)
        bias = scale * (self.conv_bias - self.running_mean) + self.scaling_factor * self.bn_bias
        
        return fused_conv_bn_scale.fused_conv_bn_scale_cuda(
            x,
            self.conv_weight,
            scale,
            bias,
            self.kernel_size
        )