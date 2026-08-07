import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel for conv + elementwise multiplication + LeakyReLU + GELU
fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

constexpr float GELU_COEF_A = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__global__ void fused_conv_activation_kernel(
    const float* input, const float* weight, const float* bias, const float* multiplier,
    float* output, float negative_slope,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size, int output_height, int output_width) {
    
    const int output_size = output_height * output_width;
    const int kernel_area = kernel_size * kernel_size;
    const int weight_stride = in_channels * kernel_area;
    
    int oc = blockIdx.z;
    int n = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;
    
    if (oc >= out_channels || n >= batch_size || oh >= output_height || ow >= output_width) {
        return;
    }
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh + kh - kernel_size / 2;
                int iw = ow + kw - kernel_size / 2;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = ((n * in_channels + ic) * input_height + ih) * input_width + iw;
                    int weight_idx = (oc * weight_stride) + (ic * kernel_area) + (kh * kernel_size + kw);
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    if (bias) {
        sum += bias[oc];
    }
    
    // Multiply by learnable scalar
    sum *= multiplier[oc];
    
    // Apply LeakyReLU
    sum = (sum > 0.0f) ? sum : sum * negative_slope;
    
    // Apply GELU
    int output_idx = ((n * out_channels + oc) * output_height + oh) * output_width + ow;
    output[output_idx] = gelu_approx(sum);
}

torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor multiplier, float negative_slope) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
                              input.options());
    
    const int block_x = 32;
    const int block_y = 8;
    dim3 block_dim(block_x, block_y);
    
    int grid_x = (output_height + block_y - 1) / block_y;
    int grid_y = batch_size;
    int grid_z = out_channels;
    dim3 grid_dim(grid_x, grid_y, grid_z);
    
    fused_conv_activation_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        multiplier.data_ptr<float>(), output.data_ptr<float>(),
        negative_slope,
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size, output_height, output_width);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_activation_cpp_source = """
torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor multiplier, float negative_slope);
"""

# Compile the inline CUDA code
fused_conv_activation = load_inline(
    name="fused_conv_activation",
    cpp_sources=fused_conv_activation_cpp_source,
    cuda_sources=fused_conv_activation_source,
    functions=["fused_conv_activation_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.negative_slope = 0.01
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_conv_activation

    def forward(self, x):
        return self.fused_op.fused_conv_activation_cuda(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )