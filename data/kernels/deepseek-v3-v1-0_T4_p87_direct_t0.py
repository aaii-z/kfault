import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused convolution + subtraction + Mish CUDA kernel
fused_conv_sub_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_conv_sub_mish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float sub1, float sub2,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int out_h, int out_w) {
    
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;
    
    if (n >= batch_size || oh >= out_h || oc >= out_channels) return;
    
    for (int ow = 0; ow < out_w; ow++) {
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    
                    if (ih < height && iw < width) {
                        int input_idx = n * (in_channels * height * width) + 
                                      ic * (height * width) + 
                                      ih * width + iw;
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size) + 
                                       ic * (kernel_size * kernel_size) + 
                                       kh * kernel_size + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias) {
            sum += bias[oc];
        }
        
        // Apply both subtractions and Mish activation
        sum = sum - sub1 - sub2;
        sum = mish_activation(sum);
        
        int output_idx = n * (out_channels * out_h * out_w) + 
                        oc * (out_h * out_w) + 
                        oh * out_w + ow;
        output[output_idx] = sum;
    }
}

torch::Tensor fused_conv_sub_mish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float sub1, float sub2, int stride, int padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, 
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 block_size(16, 16);
    dim3 grid_size((out_channels + block_size.x - 1) / block_size.x,
                  (out_h + block_size.y - 1) / block_size.y,
                  batch_size);
    
    fused_conv_sub_mish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), sub1, sub2,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, out_h, out_w);
    
    return output;
}
"""

fused_conv_sub_mish_cpp_source = """
torch::Tensor fused_conv_sub_mish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float sub1, float sub2, int stride, int padding);
"""

# Compile the inline CUDA code
fused_conv_sub_mish = load_inline(
    name="fused_conv_sub_mish",
    cpp_sources=fused_conv_sub_mish_cpp_source,
    cuda_sources=fused_conv_sub_mish_source,
    functions=["fused_conv_sub_mish_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.stride = 1
        self.padding = 0
        
        # Initialize weights and bias manually
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_op = fused_conv_sub_mish

    def forward(self, x):
        return self.fused_op.fused_conv_sub_mish_cuda(
            x, self.weight, self.bias, 
            self.subtract_value_1, self.subtract_value_2,
            self.stride, self.padding
        )