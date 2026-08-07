import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose3d + Clamp + Division
fused_conv_clamp_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cudnn.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_clamp_div_kernel(
    float* output, const float* input, const float* weight, const float* bias,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    float min_value, float divisor) {
    
    const int output_size = batch_size * out_channels * output_depth * output_height * output_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        // Calculate output indices
        const int n = idx / (out_channels * output_depth * output_height * output_width);
        const int remainder = idx % (out_channels * output_depth * output_height * output_width);
        const int oc = remainder / (output_depth * output_height * output_width);
        const int remainder2 = remainder % (output_depth * output_height * output_width);
        const int od = remainder2 / (output_height * output_width);
        const int remainder3 = remainder2 % (output_height * output_width);
        const int oh = remainder3 / output_width;
        const int ow = remainder3 % output_width;
        
        float sum = 0.0f;
        
        // Convolution operation
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_depth; ++kd) {
                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        const int id = od * stride_depth - padding_depth + kd;
                        const int ih = oh * stride_height - padding_height + kh;
                        const int iw = ow * stride_width - padding_width + kw;
                        
                        if (id >= 0 && id < input_depth && 
                            ih >= 0 && ih < input_height && 
                            iw >= 0 && iw < input_width) {
                            
                            const int input_idx = ((n * in_channels + ic) * input_depth + id) * input_height * input_width + 
                                                ih * input_width + iw;
                            const int weight_idx = ((oc * in_channels + ic) * kernel_depth + kd) * kernel_height * kernel_width + 
                                                kh * kernel_width + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        // Apply clamp and division
        sum = fmaxf(sum, min_value);
        sum /= divisor;
        
        output[idx] = sum;
    }
}

torch::Tensor fused_conv_transpose3d_clamp_div_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    float min_value, float divisor) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) {
        CHECK_INPUT(bias);
    }
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    
    const int out_channels = weight.size(1);
    const int kernel_depth = weight.size(2);
    const int kernel_height = weight.size(3);
    const int kernel_width = weight.size(4);
    
    // Calculate output dimensions
    const int output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth;
    const int output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height;
    const int output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width;
    
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                              input.options());
    
    const int output_size = batch_size * out_channels * output_depth * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    fused_conv_clamp_div_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), input.data_ptr<float>(), 
        weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_depth, kernel_height, kernel_width,
        stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width,
        min_value, divisor);
    
    return output;
}
"""

fused_conv_clamp_div_cpp_source = """
torch::Tensor fused_conv_transpose3d_clamp_div_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    float min_value, float divisor);
"""

# Compile the inline CUDA code
fused_conv_clamp_div = load_inline(
    name="fused_conv_clamp_div",
    cpp_sources=fused_conv_clamp_div_cpp_source,
    cuda_sources=fused_conv_clamp_div_source,
    functions=["fused_conv_transpose3d_clamp_div_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudnn"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = min_value
        self.divisor = divisor
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size, kernel_size
        ))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.bias, 0.0)
        
        self.fused_op = fused_conv_clamp_div

    def forward(self, x):
        stride_d, stride_h, stride_w = self.stride, self.stride, self.stride
        padding_d, padding_h, padding_w = self.padding, self.padding, self.padding
        
        return self.fused_op.fused_conv_transpose3d_clamp_div_cuda(
            x, self.weight, self.bias,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            self.min_value, self.divisor
        )