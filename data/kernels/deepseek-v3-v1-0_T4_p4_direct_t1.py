import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution + Mish + Mish
fused_conv_mish2_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float mish_activation(float x) {
    return x * tanh(log(1.0f + exp(x)));
}

__global__ void fused_conv_mish2_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int stride, int padding) {
    
    const int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    int n = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;
    
    if (n >= batch_size || oc >= out_channels || oh >= output_h || ow >= output_w) {
        return;
    }
    
    float sum = 0.0f;
    if (bias) {
        sum = bias[oc];
    }
    
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = ((n * in_channels + ic) * height + ih) * width + iw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Apply Mish activation twice
    float result = mish_activation(mish_activation(sum));
    
    int output_idx = ((n * out_channels + oc) * output_h + oh) * output_w + ow;
    output[output_idx] = result;
}

torch::Tensor fused_conv_mish2_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, 
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 block_size(32, 4);
    dim3 grid_size((output_h + block_size.y - 1) / block_size.y, out_channels, batch_size);
    
    fused_conv_mish2_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, stride, padding);
    
    return output;
}
"""

fused_conv_mish2_cpp_source = "torch::Tensor fused_conv_mish2_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"

# Compile the inline CUDA code
fused_conv_mish2 = load_inline(
    name="fused_conv_mish2",
    cpp_sources=fused_conv_mish2_cpp_source,
    cuda_sources=fused_conv_mish2_source,
    functions=["fused_conv_mish2_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_op = fused_conv_mish2

    def forward(self, x):
        return self.fused_op.fused_conv_mish2_cuda(x, self.weight, self.bias, self.stride, self.padding)