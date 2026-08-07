import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + BatchNorm + Scaling
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename T>
__global__ void fused_conv_bn_scale_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    const T* running_mean,
    const T* running_var,
    T* output,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int height,
    const int width,
    const T eps,
    const T scaling_factor) {
    
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.z * blockDim.z + threadIdx.z;
    const int c_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c_out >= out_channels || h >= height || w >= width) return;

    const int input_size = height * width;
    const int kernel_elements = kernel_size * kernel_size;
    const int output_size = height * width;

    T mean = running_mean[c_out];
    T var = running_var[c_out];
    T weight_scale = scaling_factor / sqrt(var + eps);

    T value = 0;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h + kh - kernel_size / 2;
                int w_in = w + kw - kernel_size / 2;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = c_in * input_size + h_in * width + w_in;
                    int weight_idx = c_out * (in_channels * kernel_elements) + 
                                   c_in * kernel_elements + 
                                   kh * kernel_size + kw;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    value = (value + bias[c_out] - mean) * weight_scale;
    int output_idx = c_out * output_size + h * width + w;
    output[output_idx] = value;
}

torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    double scaling_factor) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = sqrt(weight.size(1) / in_channels);

    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                              torch::dtype(input.dtype()).device(input.device()));

    const dim3 blocks(
        (out_channels + 7) / 8,
        (height + 7) / 8,
        (width + 7) / 8
    );
    const dim3 threads(8, 8, 8);

    for (int b = 0; b < batch_size; ++b) {
        fused_conv_bn_scale_kernel<float><<<blocks, threads>>>(
            input[b].data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            output[b].data_ptr<float>(),
            in_channels,
            out_channels,
            kernel_size,
            height,
            width,
            static_cast<float>(eps),
            static_cast<float>(scaling_factor)
        );
    }
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    double scaling_factor);
"""

# Load the custom CUDA extension
fused_conv_bn_scale = load_inline(
    name='fused_conv_bn_scale',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_conv_bn_scale_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        self.running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        self.scaling_factor = scaling_factor
        self.eps = 1e-5
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.conv_weight, a=sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.conv_bias, -bound, bound)

    def forward(self, x):
        return fused_conv_bn_scale.fused_conv_bn_scale_cuda(
            x,
            self.conv_weight,
            self.conv_bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scaling_factor
        )