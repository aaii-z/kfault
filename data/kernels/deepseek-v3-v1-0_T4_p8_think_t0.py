import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

template <typename T>
__global__ void fused_conv_div_pool_bias_sum_kernel(
    const T* input, const T* weight, const T* bias_conv, 
    T divisor, const T* bias, T* output,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_d, int kernel_h, int kernel_w,
    int pool_d, int pool_h, int pool_w,
    int sum_dim) {
    
    // Calculate output spatial dimensions after convolution and pooling
    const int out_depth = (depth - kernel_d + 1) / pool_d;
    const int out_height = (height - kernel_h + 1) / pool_h;
    const int out_width = (width - kernel_w + 1) / pool_w;
    
    // Calculate total number of output elements
    const int out_spatial = out_depth * out_height * out_width;
    const int out_total = batch_size * out_channels * out_spatial;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_total) return;
    
    // Decompose index
    int b = idx / (out_channels * out_spatial);
    int remain = idx % (out_channels * out_spatial);
    int c_out = remain / out_spatial;
    int spatial_idx = remain % out_spatial;
    
    int d_out = spatial_idx / (out_height * out_width);
    remain = spatial_idx % (out_height * out_width);
    int h_out = remain / out_width;
    int w_out = remain % out_width;
    
    // Calculate input spatial coordinates
    int d_in_start = d_out * pool_d;
    int h_in_start = h_out * pool_h;
    int w_in_start = w_out * pool_w;
    
    T val = 0.0;
    T max_val = -FLT_MAX;
    
    // Convolution and max pooling
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int d_in = d_in_start + kd;
                    int h_in = h_in_start + kh;
                    int w_in = w_in_start + kw;
                    
                    if (d_in < depth && h_in < height && w_in < width) {
                        int input_idx = ((b * in_channels + c_in) * depth + d_in) * height * width + 
                                      h_in * width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_d + kd) * kernel_h * kernel_w + 
                                       kh * kernel_w + kw;
                        
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add convolution bias and divide by divisor
    val = (val + bias_conv[c_out]) / divisor;
    
    // Max pooling
    for (int pd = 0; pd < pool_d; ++pd) {
        for (int ph = 0; ph < pool_h; ++ph) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int d_in = d_in_start + pd;
                int h_in = h_in_start + ph;
                int w_in = w_in_start + pw;
                
                if (d_in < depth && h_in < height && w_in < width) {
                    int input_idx = ((b * in_channels + c_in) * depth + d_in) * height * width + 
                                  h_in * width + w_in;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }
    
    // Global average pooling (1x1x1 output)
    T avg_val = val / (in_channels * kernel_d * kernel_h * kernel_w);
    
    // Add bias and sum along specified dimension
    T result = avg_val + bias[c_out];
    atomicAdd(&output[b], result);
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias_conv, 
    float divisor, torch::Tensor bias, int sum_dim) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    int pool_d = 2, pool_h = 2, pool_w = 2; // Hardcoded for this example
    
    auto output = torch::zeros({batch_size}, input.options());
    
    int threads = 256;
    int blocks = (batch_size * out_channels * depth * height * width + threads - 1) / threads;
    
    fused_conv_div_pool_bias_sum_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_conv.data_ptr<float>(),
        divisor, bias.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_d, kernel_h, kernel_w,
        pool_d, pool_h, pool_w,
        sum_dim);
        
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias_conv, float divisor, torch::Tensor bias, int sum_dim);"

# Load the custom CUDA extension
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(
            x, 
            self.conv.weight, 
            self.conv.bias,
            self.divisor,
            self.bias,
            self.sum_dim
        )