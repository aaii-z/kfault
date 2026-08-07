import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv_transpose + min + sum + GELU + bias
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
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
    
    const int output_size = batch_size * output_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= output_size) return;
    
    const int n = idx / output_width;
    const int w = idx % output_width;
    
    float sum_val = 0.0f;
    
    // For each output channel, compute conv transpose, find min across channels, then sum across height
    for (int h = 0; h < output_height; ++h) {
        float min_val = FLT_MAX;
        
        for (int c = 0; c < out_channels; ++c) {
            float conv_result = 0.0f;
            
            // Compute conv transpose for this position
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
                            
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            // Update min value across channels
            if (conv_result < min_val) {
                min_val = conv_result;
            }
        }
        
        // Sum across height dimension
        sum_val += min_val;
    }
    
    // Apply GELU and bias
    output[idx] = gelu_activation(sum_val) + bias[0];
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
    
    // Allocate output tensor
    auto output = torch::zeros({batch_size, output_width}, input.options());
    
    // Launch fused kernel
    const int output_size = batch_size * output_width;
    const int block_size = 256;
    const int num_blocks = (output_size + block_size - 1) / block_size;
    
    fused_conv_transpose_min_sum_gelu_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size, stride, padding, output_padding
    );
    
    // Reshape to match original output shape (batch_size, 1, 1, output_width)
    return output.view({batch_size, 1, 1, output_width});
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
        self.bias = nn.Parameter(torch.randn(1))
        
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias,
            self.kernel_size, self.stride, self.padding, self.output_padding
        )