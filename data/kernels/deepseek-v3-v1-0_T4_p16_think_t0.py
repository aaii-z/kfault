import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for transposed convolution + Mish + add + Hardtanh + scale
fused_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float mish(float x) {
    return x * tanh(log1pf(expf(x)));
}

__global__ void fused_conv_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const float add_value,
    const float scale
) {
    const int output_size = batch_size * out_channels * output_height * output_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size) {
        const int n = idx / (out_channels * output_height * output_width);
        const int c = (idx / (output_height * output_width)) % out_channels;
        const int h = (idx / output_width) % output_height;
        const int w = idx % output_width;
        
        float value = 0.0f;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int input_h = h + padding - kh - (kh - 1) / (kernel_size - 1);
                    const int input_w = w + padding - kw - (kw - 1) / (kernel_size - 1);
                    
                    if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                        const int input_idx = ((n * in_channels + ic) * input_height + input_h) * input_width + input_w;
                        const int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias) {
            value += bias[c];
        }
        
        // Mish activation
        value = mish(value);
        // Add value
        value += add_value;
        // Hardtanh activation
        value = fmaxf(-1.0f, fminf(1.0f, value));
        // Scale
        value *= scale;
        
        output[idx] = value;
    }
}

torch::Tensor fused_conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    float add_value,
    float scale
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int output_height = (input_height - 1) * stride + kernel_size - 2 * padding + output_padding;
    const int output_width = (input_width - 1) * stride + kernel_size - 2 * padding + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    
    const int output_size = batch_size * out_channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (output_size + threads - 1) / threads;
    
    fused_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        add_value,
        scale
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_forward", &fused_conv_forward, "Fused ConvTranspose2d with Mish, Add, Hardtanh and Scale");
}
"""

fused_conv_cpp_source = """
torch::Tensor fused_conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    float add_value,
    float scale
);
"""

# Load the custom CUDA extension
fused_conv_ext = load_inline(
    name="fused_conv",
    cpp_sources=fused_conv_cpp_source,
    cuda_sources=fused_conv_source,
    functions=["fused_conv_forward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudnn"]
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
        
        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x):
        return fused_conv_ext.fused_conv_forward(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.add_value, 
            self.scale
        )