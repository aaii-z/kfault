import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for transposed convolution + softmax + bias + scaling + sigmoid
fused_conv_softmax_bias_scale_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_softmax_bias_scale_sigmoid_kernel(
    const float* input, const float* weight, const float* bias, 
    float* output, float scaling_factor,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_size, int stride, int padding, int output_padding) {
    
    const int output_size = batch_size * out_channels * output_height * output_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= output_size) return;
    
    // Decompose global index
    const int n = idx / (out_channels * output_height * output_width);
    const int c = (idx % (out_channels * output_height * output_width)) / (output_height * output_width);
    const int h = (idx % (output_height * output_width)) / output_width;
    const int w = idx % output_width;
    
    float conv_val = 0.0f;
    
    // Compute transposed convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int input_h = h * stride - padding + kh;
                const int input_w = w * stride - padding + kw;
                
                if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                    const int input_idx = ((n * in_channels + ic) * input_height + input_h) * input_width + input_w;
                    const int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    
                    conv_val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Store convolution result temporarily
    output[idx] = conv_val;
}

__global__ void softmax_bias_scale_sigmoid_kernel(
    float* data, const float* bias, float scaling_factor,
    int batch_size, int out_channels, int output_height, int output_width) {
    
    const int spatial_size = output_height * output_width;
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int w = threadIdx.x;
    
    if (n >= batch_size || h >= output_height || w >= output_width) return;
    
    // Compute softmax across channels for each spatial position
    for (int spatial_idx = h * output_width + w; spatial_idx < spatial_size; spatial_idx += output_height * output_width) {
        // Find max value for softmax stability
        float max_val = -INFINITY;
        for (int c = 0; c < out_channels; ++c) {
            const int idx = (n * out_channels + c) * output_height * output_width + spatial_idx;
            if (data[idx] > max_val) {
                max_val = data[idx];
            }
        }
        
        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int c = 0; c < out_channels; ++c) {
            const int idx = (n * out_channels + c) * output_height * output_width + spatial_idx;
            data[idx] = expf(data[idx] - max_val);
            sum_exp += data[idx];
        }
        
        // Apply softmax, bias, scaling, and sigmoid
        for (int c = 0; c < out_channels; ++c) {
            const int idx = (n * out_channels + c) * output_height * output_width + spatial_idx;
            float softmax_val = data[idx] / sum_exp;
            float biased_val = softmax_val + bias[c];
            float scaled_val = biased_val * scaling_factor;
            data[idx] = 1.0f / (1.0f + expf(-scaled_val));
        }
    }
}

torch::Tensor fused_conv_softmax_bias_scale_sigmoid_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    float scaling_factor, int kernel_size, int stride, 
    int padding, int output_padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(0);
    
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    
    // Launch convolution kernel
    const int conv_output_size = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks_conv = (conv_output_size + block_size - 1) / block_size;
    
    fused_conv_softmax_bias_scale_sigmoid_kernel<<<num_blocks_conv, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), scaling_factor,
        batch_size, in_channels, out_channels,
        input_height, input_width, output_height, output_width,
        kernel_size, stride, padding, output_padding);
    
    // Launch softmax + bias + scale + sigmoid kernel
    dim3 grid_dims(batch_size, output_height);
    dim3 block_dims(output_width);
    
    softmax_bias_scale_sigmoid_kernel<<<grid_dims, block_dims>>>(
        output.data_ptr<float>(), bias.data_ptr<float>(), scaling_factor,
        batch_size, out_channels, output_height, output_width);
    
    return output;
}
"""

fused_conv_softmax_bias_scale_sigmoid_cpp_source = "torch::Tensor fused_conv_softmax_bias_scale_sigmoid_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor, int kernel_size, int stride, int padding, int output_padding);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_conv_softmax_bias_scale_sigmoid",
    cpp_sources=fused_conv_softmax_bias_scale_sigmoid_cpp_source,
    cuda_sources=fused_conv_softmax_bias_scale_sigmoid_source,
    functions=["fused_conv_softmax_bias_scale_sigmoid_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_conv_softmax_bias_scale_sigmoid_cuda(
            x, self.weight, self.bias, self.scaling_factor, 
            self.kernel_size, self.stride, self.padding, self.output_padding
        )