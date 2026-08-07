import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv_transpose + add + min + gelu + multiply
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr float GELU_COEF_A = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__global__ void fused_conv_transpose_ops_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float add_value, float multiply_value,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_size, int stride, int padding, int output_padding, int groups, int dilation) {
    
    int output_size = batch_size * out_channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        // Calculate output indices
        int n = idx / (out_channels * output_height * output_width);
        int remainder = idx % (out_channels * output_height * output_width);
        int c_out = remainder / (output_height * output_width);
        remainder %= (output_height * output_width);
        int h_out = remainder / output_width;
        int w_out = remainder % output_width;
        
        float result = 0.0f;
        
        // Perform transposed convolution
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in = (h_out - kh * dilation + padding) / stride;
                    int w_in = (w_out - kw * dilation + padding) / stride;
                    
                    if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width && 
                        (h_out - kh * dilation + padding) % stride == 0 && (w_out - kw * dilation + padding) % stride == 0) {
                        
                        int input_idx = n * (in_channels * input_height * input_width) + 
                                      c_in * (input_height * input_width) + 
                                      h_in * input_width + w_in;
                        
                        int weight_idx = c_in * (out_channels * kernel_size * kernel_size) + 
                                       c_out * (kernel_size * kernel_size) + 
                                       kh * kernel_size + kw;
                        
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        if (bias) {
            result += bias[c_out];
        }
        
        // Add value
        result += add_value;
        
        // Minimum with 0.0
        result = fminf(result, 0.0f);
        
        // GELU activation
        result = gelu_activation(result);
        
        // Multiply value
        result *= multiply_value;
        
        output[idx] = result;
    }
}

torch::Tensor fused_conv_transpose_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float add_value, float multiply_value, int stride, int padding,
    int output_padding, int groups, int dilation) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto out_channels = weight.size(1);
    auto kernel_size = weight.size(2);
    
    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    
    int output_size = batch_size * out_channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    fused_conv_transpose_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), add_value, multiply_value,
        batch_size, in_channels, out_channels,
        input_height, input_width, output_height, output_width,
        kernel_size, stride, padding, output_padding, groups, dilation);
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_conv_transpose_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float add_value, float multiply_value, int stride, int padding,
    int output_padding, int groups, int dilation);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_transpose_ops_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.add_value = add_value
        self.multiply_value = multiply_value
        
        # Initialize weights and bias for transposed convolution
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Default parameters for transposed convolution
        self.padding = (kernel_size - 1) // 2
        self.output_padding = stride - 1
        self.groups = 1
        self.dilation = 1
        
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_conv_transpose_ops_cuda(
            x, self.weight, self.bias, self.add_value, self.multiply_value,
            self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )