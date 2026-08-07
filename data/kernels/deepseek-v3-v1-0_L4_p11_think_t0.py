import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose2d + BatchNorm2d + Tanh
fused_conv_bn_tanh_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_transpose_bn_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int output_size = output_height * output_width;
    const int kernel_area = kernel_size * kernel_size;
    const int output_elements = batch_size * out_channels * output_size;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_elements) return;

    int n = idx / (out_channels * output_size);
    int rest = idx % (out_channels * output_size);
    int c_out = rest / output_size;
    int pos = rest % output_size;
    int h_out = pos / output_width;
    int w_out = pos % output_width;

    float value = 0.0f;

    for (int k_h = 0; k_h < kernel_size; k_h++) {
        for (int k_w = 0; k_w < kernel_size; k_w++) {
            int h_in = (h_out - k_h + padding) / stride;
            int w_in = (w_out - k_w + padding) / stride;
            
            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width &&
                (h_out - k_h + padding) % stride == 0 && 
                (w_out - k_w + padding) % stride == 0) {
                
                for (int c_in = 0; c_in < in_channels; c_in++) {
                    int input_idx = n * (in_channels * input_height * input_width) +
                                   c_in * (input_height * input_width) +
                                   h_in * input_width + w_in;
                    
                    int weight_idx = c_in * (out_channels * kernel_area) +
                                    c_out * kernel_area +
                                    k_h * kernel_size + k_w;
                    
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    value = value + bias[c_out];
    value = tanhf(value);
    
    output[idx] = value;
}

torch::Tensor fused_conv_transpose_bn_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int output_height,
    int output_width
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(1);
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    
    const int output_elements = batch_size * out_channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (output_elements + threads - 1) / threads;
    
    fused_conv_transpose_bn_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        input_height,
        input_width,
        output_height,
        output_width
    );
    
    return output;
}
"""

fused_conv_bn_tanh_cpp_source = """
torch::Tensor fused_conv_transpose_bn_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int output_height,
    int output_width
);
"""

# Compile the inline CUDA code
fused_conv_bn_tanh = load_inline(
    name="fused_conv_bn_tanh",
    cpp_sources=fused_conv_bn_tanh_cpp_source,
    cuda_sources=fused_conv_bn_tanh_source,
    functions=["fused_conv_transpose_bn_tanh_cuda"],
    verbose=True
)

class FusedConvTransposeBnTanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(FusedConvTransposeBnTanh, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size
        
        return fused_conv_bn_tanh.fused_conv_transpose_bn_tanh_cuda(
            x, self.weight, self.bias, self.kernel_size, self.stride, self.padding,
            output_height, output_width
        )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.fused_conv_bn_tanh = FusedConvTransposeBnTanh(in_channels, out_channels, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.fused_conv_bn_tanh(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x