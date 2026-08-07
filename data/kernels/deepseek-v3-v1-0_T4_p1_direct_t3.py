import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution + ReLU + bias
conv_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_relu_bias_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int output_height, int output_width) {
    
    int output_size = batch_size * out_channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        // Calculate output indices
        int n = idx / (out_channels * output_height * output_width);
        int remainder = idx % (out_channels * output_height * output_width);
        int c = remainder / (output_height * output_width);
        remainder = remainder % (output_height * output_width);
        int h = remainder / output_width;
        int w = remainder % output_width;
        
        float sum = 0.0f;
        
        // Perform convolution
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int input_h = h + kh;
                    int input_w = w + kw;
                    
                    if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                        int input_idx = n * (in_channels * height * width) + 
                                      ic * (height * width) + 
                                      input_h * width + input_w;
                        int weight_idx = c * (in_channels * kernel_size * kernel_size) + 
                                       ic * (kernel_size * kernel_size) + 
                                       kh * kernel_size + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Apply ReLU and add bias
        float result = fmaxf(0.0f, sum) + bias[c * bias_shape[1] * bias_shape[2] + 0 * bias_shape[2] + 0];
        output[idx] = result;
    }
}

torch::Tensor conv_relu_bias_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size) {
    
    int output_height = height - kernel_size + 1;
    int output_width = width - kernel_size + 1;
    int output_size = batch_size * out_channels * output_height * output_width;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    conv_relu_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, output_height, output_width);
    
    cudaDeviceSynchronize();
    return output;
}
"""

conv_relu_bias_cpp_source = """
torch::Tensor conv_relu_bias_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size);
"""

# Compile the inline CUDA code
conv_relu_bias = load_inline(
    name="conv_relu_bias",
    cpp_sources=conv_relu_bias_cpp_source,
    cuda_sources=conv_relu_bias_source,
    functions=["conv_relu_bias_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Initialize weights (equivalent to nn.Conv2d weights)
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ))
        
        # Store the custom operator
        self.conv_relu_bias_op = conv_relu_bias

    def forward(self, x):
        batch_size, _, height, width = x.shape
        return self.conv_relu_bias_op.conv_relu_bias_cuda(
            x, self.weight, self.bias, batch_size, 
            self.in_channels, self.out_channels, 
            height, width, self.kernel_size
        )