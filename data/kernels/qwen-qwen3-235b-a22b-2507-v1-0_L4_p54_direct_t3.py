import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source for fused Conv2d + Multiply + LeakyReLU + GELU
fused_conv_mul_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_THREADS 1024

// Helper function to get 4D index from flattened index
__device__ int get_index(int n, int c, int h, int w, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

__global__ void fused_conv_mul_lrelu_gelu(const float* input,
                                          const float* weight,
                                          const float* bias,
                                          const float* multiplier,
                                          float* output,
                                          int batch_size, int in_channels, int out_channels,
                                          int input_height, int input_width,
                                          int output_height, int output_width,
                                          int kernel_size,
                                          int stride, int padding) {
    int total_elements = batch_size * out_channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;

    // Decode index
    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    // Convolution computation
    float conv_val = 0.0f;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            for (int ic = 0; ic < in_channels; ++ic) {
                int ih = h * stride + kh - padding;
                int iw = w * stride + kw - padding;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int in_idx = get_index(n, ic, ih, iw, in_channels, input_height, input_width);
                    int weight_idx = c * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + kh * kernel_size + kw;
                    conv_val += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    conv_val += bias[c];
    
    // Multiply by channel-wise multiplier (multiplier shape: [out_channels, 1, 1])
    float mul_val = conv_val * multiplier[c];
    
    // LeakyReLU: max(0.01 * x, x)
    float lrelu_val = fmaxf(mul_val * 0.01f, mul_val);
    
    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float x = lrelu_val;
    float x_cubed = x * x * x;
    float tanh_val = tanhf(0.7978845608028654f * (x + 0.044715f * x_cubed));
    float gelu_val = x * 0.5f * (1.0f + tanh_val);
    
    output[idx] = gelu_val;
}

torch::Tensor fused_conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    int padding,
    int stride) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    // Correctly compute output dimensions using PyTorch's formula
    auto output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    auto output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
                               input.options());
    
    int total_elements = batch_size * out_channels * output_height * output_width;
    int block_size = MAX_THREADS;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_conv_mul_lrelu_gelu<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size,
        stride, padding
    );
    
    return output;
}
"""

fused_conv_cpp_source = """
torch::Tensor fused_conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor multiplier,
    int padding,
    int stride);
"""

# Compile the inline CUDA kernel
fused_ops = load_inline(
    name="fused_conv_mul_activation",
    cpp_sources=fused_conv_cpp_source,
    cuda_sources=fused_conv_mul_activation_source,
    functions=["fused_conv_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for Conv2d + Multiply + LeakyReLU + GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        # Conv2d parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2  # Match PyTorch's default padding for Conv2d when kernel_size=3

        # Learnable parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(self.bias)

        # Fused CUDA function
        self.fused_conv_forward = fused_ops.fused_conv_forward

    def forward(self, x):
        return self.fused_conv_forward(x, self.weight, self.bias, self.multiplier, self.padding, self.stride)