import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv + min + add + scale
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cudnn.h>

__global__ void fused_conv_min_add_scale_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, float constant_value,
    const float* param_bias, float scaling_factor) {
    
    const int output_h = height - kernel_size + 1;
    const int output_w = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * output_h * output_w;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    
    // Calculate output indices
    int n = idx / (out_channels * output_h * output_w);
    int remainder = idx % (out_channels * output_h * output_w);
    int c = remainder / (output_h * output_w);
    remainder = remainder % (output_h * output_w);
    int h = remainder / output_w;
    int w = remainder % output_w;
    
    // Perform convolution
    float conv_result = 0.0f;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int input_h = h + kh;
                int input_w = w + kw;
                int input_idx = n * (in_channels * height * width) + 
                              ic * (height * width) + 
                              input_h * width + input_w;
                int weight_idx = c * (in_channels * kernel_size * kernel_size) + 
                               ic * (kernel_size * kernel_size) + 
                               kh * kernel_size + kw;
                
                conv_result += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Add convolution bias
    conv_result += bias[c];
    
    // Apply min with constant
    float min_result = fminf(conv_result, constant_value);
    
    // Add parameter bias (broadcasted)
    int bias_idx = c;
    float biased_result = min_result + param_bias[bias_idx];
    
    // Apply scaling
    output[idx] = biased_result * scaling_factor;
}

torch::Tensor fused_conv_min_add_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    torch::Tensor param_bias, float constant_value, float scaling_factor,
    int kernel_size) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    const int output_h = height - kernel_size + 1;
    const int output_w = width - kernel_size + 1;
    const int output_size = batch_size * out_channels * output_h * output_w;
    
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, 
                              input.options());
    
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    fused_conv_min_add_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), conv_bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, constant_value,
        param_bias.data_ptr<float>(), scaling_factor);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_conv_min_add_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    torch::Tensor param_bias, float constant_value, float scaling_factor,
    int kernel_size);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_min_add_scale_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        self.constant_value = constant_value
        self.param_bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.kernel_size = kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_conv_min_add_scale_cuda(
            x, self.conv_weight, self.conv_bias, self.param_bias,
            self.constant_value, self.scaling_factor, self.kernel_size
        )