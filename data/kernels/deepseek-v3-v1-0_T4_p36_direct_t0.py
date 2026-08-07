import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv_transpose + min + sum + GELU + bias
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define GELU_SCALING_FACTOR 0.044715f
#define SQRT_2_OVER_PI 0.7978845608028654f

__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_SCALING_FACTOR * x * x)));
}

__global__ void fused_conv_transpose_min_sum_gelu_bias_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, 
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_size, int stride, int padding, int output_padding) {
    
    const int output_size = batch_size * out_channels * output_height * output_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= output_size) return;
    
    // Decompose output index
    const int n = idx / (out_channels * output_height * output_width);
    const int c = (idx / (output_height * output_width)) % out_channels;
    const int h = (idx / output_width) % output_height;
    const int w = idx % output_width;
    
    float result = 0.0f;
    
    // Compute conv transpose operation
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int input_h = (h - kh + padding) / stride;
                int input_w = (w - kw + padding) / stride;
                
                if ((h - kh + padding) % stride == 0 && 
                    (w - kw + padding) % stride == 0 &&
                    input_h >= 0 && input_h < input_height &&
                    input_w >= 0 && input_w < input_width) {
                    
                    int input_idx = n * (in_channels * input_height * input_width) + 
                                  ic * (input_height * input_width) + 
                                  input_h * input_width + input_w;
                    
                    int weight_idx = ic * (out_channels * kernel_size * kernel_size) + 
                                   c * (kernel_size * kernel_size) + 
                                   kh * kernel_size + kw;
                    
                    result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Store intermediate result
    output[idx] = result;
}

__global__ void min_reduce_kernel(const float* input, float* min_output, 
                                 int batch_size, int out_channels, 
                                 int output_height, int output_width) {
    const int spatial_size = output_height * output_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * spatial_size) return;
    
    const int n = idx / spatial_size;
    const int spatial_idx = idx % spatial_size;
    
    float min_val = input[n * (out_channels * spatial_size) + spatial_idx];
    for (int c = 1; c < out_channels; ++c) {
        float current = input[n * (out_channels * spatial_size) + c * spatial_size + spatial_idx];
        if (current < min_val) {
            min_val = current;
        }
    }
    min_output[n * spatial_size + spatial_idx] = min_val;
}

__global__ void sum_reduce_kernel(const float* input, float* sum_output,
                                 int batch_size, int output_height, int output_width) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * output_width) return;
    
    const int n = idx / output_width;
    const int w = idx % output_width;
    
    float sum_val = 0.0f;
    for (int h = 0; h < output_height; ++h) {
        sum_val += input[n * (output_height * output_width) + h * output_width + w];
    }
    sum_output[n * output_width + w] = sum_val;
}

__global__ void gelu_bias_kernel(float* data, const float* bias, 
                                int batch_size, int output_width) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * output_width) return;
    
    const int n = idx / output_width;
    const int w = idx % output_width;
    
    data[idx] = gelu_activation(data[idx]) + bias[w];
}

torch::Tensor fused_operator_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                 int kernel_size, int stride, int padding, int output_padding) {
    
    // Get input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    
    int out_channels = weight.size(1);
    int output_height = (input_height - 1) * stride + kernel_size - 2 * padding + output_padding;
    int output_width = (input_width - 1) * stride + kernel_size - 2 * padding + output_padding;
    
    // Allocate intermediate and output tensors
    auto conv_output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());
    auto min_output = torch::zeros({batch_size, output_height, output_width}, input.options());
    auto sum_output = torch::zeros({batch_size, output_width}, input.options());
    auto final_output = torch::zeros({batch_size, output_width}, input.options());
    
    // Launch conv transpose kernel
    const int conv_size = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int conv_blocks = (conv_size + block_size - 1) / block_size;
    
    fused_conv_transpose_min_sum_gelu_bias_kernel<<<conv_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        conv_output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size, stride, padding, output_padding
    );
    
    // Launch min reduction kernel
    const int min_size = batch_size * output_height * output_width;
    const int min_blocks = (min_size + block_size - 1) / block_size;
    
    min_reduce_kernel<<<min_blocks, block_size>>>(
        conv_output.data_ptr<float>(), min_output.data_ptr<float>(),
        batch_size, out_channels, output_height, output_width
    );
    
    // Launch sum reduction kernel
    const int sum_size = batch_size * output_width;
    const int sum_blocks = (sum_size + block_size - 1) / block_size;
    
    sum_reduce_kernel<<<sum_blocks, block_size>>>(
        min_output.data_ptr<float>(), sum_output.data_ptr<float>(),
        batch_size, output_height, output_width
    );
    
    // Launch GELU + bias kernel
    const int gelu_size = batch_size * output_width;
    const int gelu_blocks = (gelu_size + block_size - 1) / block_size;
    
    gelu_bias_kernel<<<gelu_blocks, block_size>>>(
        sum_output.data_ptr<float>(), bias.data_ptr<float>(),
        batch_size, output_width
    );
    
    // Reshape to match original output shape
    return sum_output.view({batch_size, 1, 1, output_width});
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                 int kernel_size, int stride, int padding, int output_padding);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape[2]))  # Only need the width dimension
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias,
            self.kernel_size, self.stride, self.padding, self.output_padding
        )