import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Conv3D + Division + MaxPool3D
fused_conv_div_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define CUDA_CHECK(err) do { if (err != cudaSuccess) { printf("CUDA error: %s\\n", cudaGetErrorString(err)); } } while (0)

template<typename T>
__global__ void fused_conv_div_pool_kernel(
    const T* input, const T* weight, const T* bias,
    T* output, int batch_size, int in_channels, int out_channels,
    int depth, int height, int width, int kernel_d, int kernel_h, int kernel_w,
    T divisor, int pool_d, int pool_h, int pool_w) {
    
    const int output_d = (depth - kernel_d + 1) / pool_d;
    const int output_h = (height - kernel_h + 1) / pool_h;
    const int output_w = (width - kernel_w + 1) / pool_w;
    
    const int total_output_size = batch_size * out_channels * output_d * output_h * output_w;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_output_size) return;
    
    // Decompose index
    int n = idx / (out_channels * output_d * output_h * output_w);
    int remainder = idx % (out_channels * output_d * output_h * output_w);
    int c = remainder / (output_d * output_h * output_w);
    remainder %= (output_d * output_h * output_w);
    int d = remainder / (output_h * output_w);
    remainder %= (output_h * output_w);
    int h = remainder / output_w;
    int w = remainder % output_w;
    
    // Original spatial positions (before pooling)
    int start_d = d * pool_d;
    int start_h = h * pool_h;
    int start_w = w * pool_w;
    
    T max_val = -FLT_MAX;
    
    // Process each pooling region
    for (int pd = 0; pd < pool_d; pd++) {
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; pw++) {
                int current_d = start_d + pd;
                int current_h = start_h + ph;
                int current_w = start_w + pw;
                
                if (current_d < depth - kernel_d + 1 && current_h < height - kernel_h + 1 && current_w < width - kernel_w + 1) {
                    T conv_val = bias ? bias[c] : 0.0f;
                    
                    // Convolution operation
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kd = 0; kd < kernel_d; kd++) {
                            for (int kh = 0; kh < kernel_h; kh++) {
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    int input_d = current_d + kd;
                                    int input_h = current_h + kh;
                                    int input_w = current_w + kw;
                                    
                                    int input_idx = ((n * in_channels + ic) * depth + input_d) * height * width + input_h * width + input_w;
                                    int weight_idx = ((c * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                                    
                                    conv_val += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    
                    // Division and max pooling
                    T divided_val = conv_val / divisor;
                    max_val = max(max_val, divided_val);
                }
            }
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor fused_conv_div_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float divisor, std::vector<int> kernel_size, std::vector<int> pool_size) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(0);
    
    int kernel_d = kernel_size[0], kernel_h = kernel_size[1], kernel_w = kernel_size[2];
    int pool_d = pool_size[0], pool_h = pool_size[1], pool_w = pool_size[2];
    
    int output_d = (depth - kernel_d + 1) / pool_d;
    int output_h = (height - kernel_h + 1) / pool_h;
    int output_w = (width - kernel_w + 1) / pool_w;
    
    auto output = torch::zeros({batch_size, out_channels, output_d, output_h, output_w}, input.options());
    
    int total_size = batch_size * out_channels * output_d * output_h * output_w;
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_conv_div_pool_kernel<float><<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        depth, height, width, kernel_d, kernel_h, kernel_w,
        divisor, pool_d, pool_h, pool_w);
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_conv_div_pool_cpp_source = """
torch::Tensor fused_conv_div_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float divisor, std::vector<int> kernel_size, std::vector<int> pool_size);
"""

# Define the global average pooling + bias + sum kernel
global_avg_bias_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_bias_sum_kernel(
    const float* input, const float* bias, float* output,
    int batch_size, int channels, int depth, int height, int width, int sum_dim) {
    
    const int spatial_size = depth * height * width;
    const int total_size = batch_size * channels;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_size) return;
    
    int n = idx / channels;
    int c = idx % channels;
    
    // Global average pooling
    float sum = 0.0f;
    for (int i = 0; i < spatial_size; i++) {
        sum += input[n * channels * spatial_size + c * spatial_size + i];
    }
    float avg_val = sum / spatial_size;
    
    // Add bias
    avg_val += bias[c];
    
    // Store result
    output[n * channels + c] = avg_val;
}

__global__ void sum_reduction_kernel(
    const float* input, float* output, int batch_size, int channels, int sum_dim) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    float sum = 0.0f;
    for (int c = 0; c < channels; c++) {
        sum += input[idx * channels + c];
    }
    output[idx] = sum;
}

torch::Tensor global_avg_bias_sum_cuda(torch::Tensor input, torch::Tensor bias, int sum_dim) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    // Global average pooling + bias
    auto intermediate = torch::zeros({batch_size, channels}, input.options());
    
    int total_size = batch_size * channels;
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    
    global_avg_bias_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), intermediate.data_ptr<float>(),
        batch_size, channels, depth, height, width, sum_dim);
    
    // Sum along specified dimension
    if (sum_dim == 1) {
        auto output = torch::zeros({batch_size}, input.options());
        num_blocks = (batch_size + block_size - 1) / block_size;
        sum_reduction_kernel<<<num_blocks, block_size>>>(
            intermediate.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, sum_dim);
        return output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1);
    }
    
    return intermediate;
}
"""

global_avg_bias_sum_cpp_source = """
torch::Tensor global_avg_bias_sum_cuda(torch::Tensor input, torch::Tensor bias, int sum_dim);
"""

# Compile the inline CUDA code
fused_conv_div_pool = load_inline(
    name="fused_conv_div_pool",
    cpp_sources=fused_conv_div_pool_cpp_source,
    cuda_sources=fused_conv_div_pool_source,
    functions=["fused_conv_div_pool_cuda"],
    verbose=False
)

global_avg_bias_sum = load_inline(
    name="global_avg_bias_sum",
    cpp_sources=global_avg_bias_sum_cpp_source,
    cuda_sources=global_avg_bias_sum_source,
    functions=["global_avg_bias_sum_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.divisor = divisor
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.sum_dim = sum_dim
        
        # Initialize convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias_conv = nn.Parameter(torch.randn(out_channels))
        self.bias_add = nn.Parameter(torch.randn(bias_shape))
        
        self.fused_conv_div_pool = fused_conv_div_pool
        self.global_avg_bias_sum = global_avg_bias_sum

    def forward(self, x):
        # Fused: Conv3D + Division + MaxPool3D
        x = self.fused_conv_div_pool.fused_conv_div_pool_cuda(
            x, self.weight, self.bias_conv, 
            self.divisor, list(self.kernel_size), list(self.pool_size)
        )
        
        # Fused: GlobalAvgPool3D + Bias addition + Sum along dimension
        x = self.global_avg_bias_sum.global_avg_bias_sum_cuda(x, self.bias_add.squeeze(), self.sum_dim)
        
        return x