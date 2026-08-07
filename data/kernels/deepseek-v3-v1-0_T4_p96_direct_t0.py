import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + Scale + MaxPool3d + Clamp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#define CUDA_CHECK(call) {\\
    cudaError_t err = call;\\
    if (err != cudaSuccess) {\\
        fprintf(stderr, "CUDA error at %s:%d: %s\\\\n", __FILE__, __LINE__, cudaGetErrorString(err));\\
        exit(EXIT_FAILURE);\\
    }\\
}

__device__ __forceinline__ float max3(float a, float b, float c) {
    return fmaxf(fmaxf(a, b), c);
}

__global__ void fused_conv_transpose_scale_maxpool_clamp_kernel(
    const float* input, float* output,
    const float* weight, const float* bias,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width,
    int kernel_size, int stride, int padding,
    float scale, int maxpool_kernel_size,
    float clamp_min, float clamp_max) {
    
    // Calculate output dimensions after maxpool
    int pooled_depth = (output_depth + maxpool_kernel_size - 1) / maxpool_kernel_size;
    int pooled_height = (output_height + maxpool_kernel_size - 1) / maxpool_kernel_size;
    int pooled_width = (output_width + maxpool_kernel_size - 1) / maxpool_kernel_size;
    
    // Global thread index for output tensor (after maxpool)
    int n = blockIdx.x;
    int c = blockIdx.y;
    int d = threadIdx.x * blockDim.y + threadIdx.y;
    int h = blockIdx.z % pooled_height;
    int w = blockIdx.z / pooled_height;
    
    if (n >= batch_size || c >= out_channels || d >= pooled_depth || h >= pooled_height || w >= pooled_width) {
        return;
    }
    
    // Input coordinates for the maxpool window
    int start_d = d * maxpool_kernel_size;
    int start_h = h * maxpool_kernel_size;
    int start_w = w * maxpool_kernel_size;
    int end_d = min(start_d + maxpool_kernel_size, output_depth);
    int end_h = min(start_h + maxpool_kernel_size, output_height);
    int end_w = min(start_w + maxpool_kernel_size, output_width);
    
    float max_val = -FLT_MAX;
    
    // Iterate over maxpool window
    for (int pd = start_d; pd < end_d; ++pd) {
        for (int ph = start_h; ph < end_h; ++ph) {
            for (int pw = start_w; pw < end_w; ++pw) {
                
                float conv_val = 0.0f;
                
                // Convolution operation
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kd = 0; kd < kernel_size; ++kd) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                
                                // Calculate input coordinates
                                int input_d = (pd - kd + padding) / stride;
                                int input_h = (ph - kh + padding) / stride;
                                int input_w = (pw - kw + padding) / stride;
                                
                                if (input_d >= 0 && input_d < input_depth &&
                                    input_h >= 0 && input_h < input_height &&
                                    input_w >= 0 && input_w < input_width &&
                                    (pd - kd + padding) % stride == 0 &&
                                    (ph - kh + padding) % stride == 0 &&
                                    (pw - kw + padding) % stride == 0) {
                                    
                                    int input_idx = ((n * in_channels + ic) * input_depth + input_d) * input_height * input_width +
                                                   input_h * input_width + input_w;
                                    
                                    int weight_idx = ((c * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size +
                                                     kh * kernel_size + kw;
                                    
                                    conv_val += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                // Add bias
                if (bias) {
                    conv_val += bias[c];
                }
                
                // Apply scale
                conv_val *= scale;
                
                // Apply clamp
                conv_val = fminf(fmaxf(conv_val, clamp_min), clamp_max);
                
                // Update max value
                max_val = fmaxf(max_val, conv_val);
            }
        }
    }
    
    // Write the maxpool result
    int output_idx = ((n * out_channels + c) * pooled_depth + d) * pooled_height * pooled_width +
                     h * pooled_width + w;
    output[output_idx] = max_val;
}

torch::Tensor fused_conv_transpose_scale_maxpool_clamp_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding,
    float scale, int maxpool_kernel_size,
    float clamp_min, float clamp_max) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    
    auto out_channels = weight.size(1);
    
    // Calculate output dimensions for conv transpose
    int output_depth = (input_depth - 1) * stride + kernel_size - 2 * padding;
    int output_height = (input_height - 1) * stride + kernel_size - 2 * padding;
    int output_width = (input_width - 1) * stride + kernel_size - 2 * padding;
    
    // Calculate output dimensions after maxpool
    int pooled_depth = (output_depth + maxpool_kernel_size - 1) / maxpool_kernel_size;
    int pooled_height = (output_height + maxpool_kernel_size - 1) / maxpool_kernel_size;
    int pooled_width = (output_width + maxpool_kernel_size - 1) / maxpool_kernel_size;
    
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({batch_size, out_channels, pooled_depth, pooled_height, pooled_width}, options);
    
    // Configure kernel launch parameters
    dim3 blocks(batch_size, out_channels, pooled_height * pooled_width);
    dim3 threads(16, 16);  // 256 threads per block
    
    fused_conv_transpose_scale_maxpool_clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_size, stride, padding,
        scale, maxpool_kernel_size,
        clamp_min, clamp_max);
    
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_conv_transpose_scale_maxpool_clamp_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int stride, int padding, float scale, int maxpool_kernel_size, float clamp_min, float clamp_max);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_transpose_scale_maxpool_clamp_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0
        self.clamp_max = 1
        
        # Initialize weights and bias for the transposed convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Global average pooling remains as is (efficient in PyTorch)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Load the custom CUDA operation
        self.fused_ops = fused_ops

    def forward(self, x):
        # Fused operation: ConvTranspose3d + Scale + MaxPool3d + Clamp
        x = self.fused_ops.fused_conv_transpose_scale_maxpool_clamp_cuda(
            x, self.weight, self.bias,
            self.kernel_size, self.stride, self.padding,
            self.scale, self.maxpool_kernel_size,
            self.clamp_min, self.clamp_max
        )
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        return x