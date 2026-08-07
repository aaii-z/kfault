import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Conv-BN-Scale CUDA kernel
fused_conv_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cassert>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_bn_scale_kernel(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var,
    const float* weight_bn, const float* bias_bn,
    float* output, float scaling_factor,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int padding, int stride) {
    
    const int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    // Calculate output position
    const int n = blockIdx.x;  // batch index
    const int c = blockIdx.y;  // output channel
    const int h = threadIdx.y; // output height
    const int w = threadIdx.x; // output width
    
    if (n >= batch_size || c >= out_channels || h >= output_h || w >= output_w) {
        return;
    }
    
    float sum = 0.0f;
    
    // Perform convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int input_h = h * stride + kh - padding;
                int input_w = w * stride + kw - padding;
                
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
    
    // Add bias if present
    if (bias) {
        sum += bias[c];
    }
    
    // Apply BatchNorm: (x - mean) / sqrt(var + eps) * weight_bn + bias_bn
    const float eps = 1e-5f;
    float bn_output = (sum - running_mean[c]) / sqrtf(running_var[c] + eps);
    bn_output = bn_output * weight_bn[c] + bias_bn[c];
    
    // Apply scaling
    output[n * (out_channels * output_h * output_w) + 
           c * (output_h * output_w) + 
           h * output_w + w] = bn_output * scaling_factor;
}

torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight_bn, torch::Tensor bias_bn,
    float scaling_factor, int padding, int stride) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);
    CHECK_INPUT(weight_bn);
    CHECK_INPUT(bias_bn);
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, 
                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    dim3 block_dim(output_w, output_h);
    dim3 grid_dim(batch_size, out_channels);
    
    fused_conv_bn_scale_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        weight_bn.data_ptr<float>(), bias_bn.data_ptr<float>(),
        output.data_ptr<float>(), scaling_factor,
        batch_size, in_channels, out_channels,
        height, width, kernel_size, padding, stride);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_bn_scale_cpp_source = """
torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight_bn, torch::Tensor bias_bn,
    float scaling_factor, int padding, int stride);
"""

# Compile the inline CUDA code
fused_conv_bn_scale = load_inline(
    name="fused_conv_bn_scale",
    cpp_sources=fused_conv_bn_scale_cpp_source,
    cuda_sources=fused_conv_bn_scale_source,
    functions=["fused_conv_bn_scale_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        self.padding = kernel_size // 2
        self.stride = 1
        
        # Initialize convolution parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize BatchNorm parameters
        self.running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        self.weight_bn = nn.Parameter(torch.ones(out_channels))
        self.bias_bn = nn.Parameter(torch.zeros(out_channels))
        
        self.fused_op = fused_conv_bn_scale

    def forward(self, x):
        return self.fused_op.fused_conv_bn_scale_cuda(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.weight_bn, self.bias_bn,
            self.scaling_factor, self.padding, self.stride
        )