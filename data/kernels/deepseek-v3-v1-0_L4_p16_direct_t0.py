import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for transposed convolution + Mish + add + Hardtanh + scale
fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ float mish_activation(float x) {
    return x * tanh(log(1.0f + exp(x)));
}

__device__ float hardtanh_activation(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__global__ void fused_conv_activation_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float add_value, float scale,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_size, int stride, int padding, int output_padding) {
    
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_size = batch_size * out_channels * output_height * output_width;
    
    if (output_idx < total_output_size) {
        // Calculate output indices
        int n = output_idx / (out_channels * output_height * output_width);
        int remainder = output_idx % (out_channels * output_height * output_width);
        int c = remainder / (output_height * output_width);
        remainder = remainder % (output_height * output_width);
        int h = remainder / output_width;
        int w = remainder % output_width;
        
        float result = 0.0f;
        
        // Transposed convolution operation
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int input_h = (h + padding - kh) / stride;
                int input_w = (w + padding - kw) / stride;
                
                if ((h + padding - kh) % stride == 0 && 
                    (w + padding - kw) % stride == 0 &&
                    input_h >= 0 && input_h < input_height &&
                    input_w >= 0 && input_w < input_width) {
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
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
        
        // Add bias
        if (bias) {
            result += bias[c];
        }
        
        // Mish activation
        result = mish_activation(result);
        
        // Add value
        result += add_value;
        
        // Hardtanh activation
        result = hardtanh_activation(result, -1.0f, 1.0f);
        
        // Scale
        result *= scale;
        
        output[output_idx] = result;
    }
}

torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float add_value, float scale,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_size, int stride, int padding, int output_padding) {
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_size = output.numel();
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_conv_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), add_value, scale,
        batch_size, in_channels, out_channels,
        input_height, input_width, output_height, output_width,
        kernel_size, stride, padding, output_padding);
    
    return output;
}
"""

fused_conv_activation_cpp_source = """
torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float add_value, float scale,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_size, int stride, int padding, int output_padding);
"""

# Compile the inline CUDA code
fused_conv_activation = load_inline(
    name="fused_conv_activation",
    cpp_sources=fused_conv_activation_cpp_source,
    cuda_sources=fused_conv_activation_source,
    functions=["fused_conv_activation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale
        
        # Initialize weights and bias for transposed convolution
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_op = fused_conv_activation

    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        return self.fused_op.fused_conv_activation_cuda(
            x, self.weight, self.bias, self.add_value, self.scale,
            batch_size, self.in_channels, self.out_channels,
            input_height, input_width, output_height, output_width,
            self.kernel_size, self.stride, self.padding, self.output_padding
        )