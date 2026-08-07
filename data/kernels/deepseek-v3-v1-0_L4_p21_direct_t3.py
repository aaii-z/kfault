import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv + bias + scale + sigmoid
fused_conv_bias_scale_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_bias_scale_sigmoid_kernel(
    const float* input, const float* weight, const float* bias, const float* scale,
    float* output, int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size) {
    
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    const int spatial_size = output_height * output_width;
    const int total_elements = batch_size * out_channels * spatial_size;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Calculate indices
    int b = idx / (out_channels * spatial_size);
    int c = (idx % (out_channels * spatial_size)) / spatial_size;
    int hw = idx % spatial_size;
    int h_out = hw / output_width;
    int w_out = hw % output_width;
    
    // Convolution
    float conv_result = 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;
                
                if (h_in < input_height && w_in < input_width) {
                    int input_idx = b * (in_channels * input_height * input_width) 
                                  + ic * (input_height * input_width) 
                                  + h_in * input_width + w_in;
                    
                    int weight_idx = c * (in_channels * kernel_size * kernel_size)
                                   + ic * (kernel_size * kernel_size)
                                   + kh * kernel_size + kw;
                    
                    conv_result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias and scale
    float result = (conv_result + bias[c]) * scale[c];
    
    // Sigmoid
    result = 1.0f / (1.0f + expf(-result));
    
    output[idx] = result;
}

torch::Tensor fused_conv_bias_scale_sigmoid_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor scale, int kernel_size) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto out_channels = weight.size(0);
    
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    
    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        scale.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, 
        input_height, input_width, kernel_size);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_bias_scale_sigmoid_cpp_source = "torch::Tensor fused_conv_bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor scale, int kernel_size);"

# Compile the inline CUDA code
fused_conv_bias_scale_sigmoid = load_inline(
    name="fused_conv_bias_scale_sigmoid",
    cpp_sources=fused_conv_bias_scale_sigmoid_cpp_source,
    cuda_sources=fused_conv_bias_scale_sigmoid_source,
    functions=["fused_conv_bias_scale_sigmoid_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.num_groups = num_groups
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.kernel_size = kernel_size
        self.fused_operator = fused_conv_bias_scale_sigmoid

    def forward(self, x):
        x = self.fused_operator.fused_conv_bias_scale_sigmoid_cuda(x, self.weight, self.bias, self.scale, self.kernel_size)
        x = self.group_norm(x)
        return x