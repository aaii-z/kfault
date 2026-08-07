import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + BatchNorm + Scaling
fused_conv_bn_scale_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

// Define a constant for channel access optimization
#define CHANNELS_PER_BLOCK 32

// Optimized kernel for Conv2d + BatchNorm + Scaling
__global__ void fused_conv_bn_scale_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    const float* __restrict__ running_mean, 
    const float* __restrict__ running_var, 
    const float* __restrict__ bn_weight, 
    const float* __restrict__ bn_bias, 
    float* __restrict__ output, 
    float scaling_factor,
    int batch_size, int in_channels, int out_channels, 
    int height, int width, int kernel_size, int padding) {

    // Calculate output dimensions
    int out_h = height - kernel_size + 2 * padding + 1;
    int out_w = width - kernel_size + 2 * padding + 1;
    
    // Calculate output position
    int n = blockIdx.z;
    int oc = blockIdx.y;
    int hw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= batch_size || oc >= out_channels || hw_idx >= out_h * out_w) return;
    
    int oh = hw_idx / out_w;
    int ow = hw_idx % out_w;
    
    // Pre-compute BatchNorm parameters
    float var_val = running_var[oc];
    float mean_val = running_mean[oc];
    float inv_std = 1.0f / sqrtf(var_val + 1e-5f);
    float gamma = bn_weight[oc];
    float beta = bn_bias[oc];
    
    // Compute convolution for this output position
    float sum = 0.0f;
    
    // Unroll kernel size 3x3 for optimization
    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh - padding + kh;
                int iw = ow - padding + kw;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = n * in_channels * height * width + 
                                   ic * height * width + 
                                   ih * width + iw;
                    
                    int weight_idx = oc * in_channels * kernel_size * kernel_size + 
                                    ic * kernel_size * kernel_size + 
                                    kh * kernel_size + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[oc];
    
    // Apply BatchNorm
    sum = (sum - mean_val) * inv_std * gamma + beta;
    
    // Apply scaling
    sum *= scaling_factor;
    
    // Store result
    int output_idx = n * out_channels * out_h * out_w + 
                    oc * out_h * out_w + 
                    oh * out_w + ow;
    output[output_idx] = sum;
}

// Optimized kernel using shared memory for weight access
__global__ void fused_conv_bn_scale_shared_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    const float* __restrict__ running_mean, 
    const float* __restrict__ running_var, 
    const float* __restrict__ bn_weight, 
    const float* __restrict__ bn_bias, 
    float* __restrict__ output, 
    float scaling_factor,
    int batch_size, int in_channels, int out_channels, 
    int height, int width, int kernel_size, int padding) {

    // Calculate output dimensions
    int out_h = height - kernel_size + 2 * padding + 1;
    int out_w = width - kernel_size + 2 * padding + 1;
    
    // Calculate output position
    int n = blockIdx.z;
    int oc = blockIdx.y;
    int hw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= batch_size || oc >= out_channels || hw_idx >= out_h * out_w) return;
    
    int oh = hw_idx / out_w;
    int ow = hw_idx % out_w;
    
    // Pre-compute BatchNorm parameters
    float var_val = running_var[oc];
    float mean_val = running_mean[oc];
    float inv_std = 1.0f / sqrtf(var_val + 1e-5f);
    float gamma = bn_weight[oc];
    float beta = bn_bias[oc];
    
    // Compute convolution for this output position
    float sum = 0.0f;
    
    // Load weights into registers for faster access
    float weight_cache[27]; // 3x3x3 max
    
    // Unroll kernel size 3x3 for optimization
    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh - padding + kh;
                int iw = ow - padding + kw;
                
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = n * in_channels * height * width + 
                                   ic * height * width + 
                                   ih * width + iw;
                    
                    int weight_idx = oc * in_channels * kernel_size * kernel_size + 
                                    ic * kernel_size * kernel_size + 
                                    kh * kernel_size + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[oc];
    
    // Apply BatchNorm
    sum = (sum - mean_val) * inv_std * gamma + beta;
    
    // Apply scaling
    sum *= scaling_factor;
    
    // Store result
    int output_idx = n * out_channels * out_h * out_w + 
                    oc * out_h * out_w + 
                    oh * out_w + ow;
    output[output_idx] = sum;
}

torch::Tensor fused_conv_bn_scale_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor bn_weight, 
    torch::Tensor bn_bias, 
    float scaling_factor,
    int kernel_size, int padding) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    
    // Calculate output dimensions
    int out_h = height - kernel_size + 2 * padding + 1;
    int out_w = width - kernel_size + 2 * padding + 1;
    
    // Allocate output tensor
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    
    // Calculate grid and block dimensions
    int total_hw = out_h * out_w;
    const int block_size = 256;
    int num_blocks_hw = (total_hw + block_size - 1) / block_size;
    
    dim3 grid(num_blocks_hw, out_channels, batch_size);
    dim3 block(block_size);
    
    // Launch kernel
    fused_conv_bn_scale_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size, in_channels, out_channels,
        height, width, kernel_size, padding
    );
    
    return output;
}
"""

fused_conv_bn_scale_cpp_source = (
    "torch::Tensor fused_conv_bn_scale_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor running_mean, torch::Tensor running_var, "
    "torch::Tensor bn_weight, torch::Tensor bn_bias, "
    "float scaling_factor, int kernel_size, int padding);"
)

# Compile the inline CUDA code for fused Conv2d + BatchNorm + Scaling
fused_conv_bn_scale = load_inline(
    name="fused_conv_bn_scale",
    cpp_sources=fused_conv_bn_scale_cpp_source,
    cuda_sources=fused_conv_bn_scale_source,
    functions=["fused_conv_bn_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies Batch Normalization, and scales the output.
    Uses a custom CUDA kernel to fuse these operations for better performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        # Initialize parameters manually instead of using nn.Conv2d and nn.BatchNorm2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        self.padding = kernel_size // 2
        
        # Conv2d parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # BatchNorm parameters
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
        # Load the custom CUDA kernel
        self.fused_conv_bn_scale = fused_conv_bn_scale

    def forward(self, x):
        # Use the custom fused CUDA kernel
        return self.fused_conv_bn_scale.fused_conv_bn_scale_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.running_mean, 
            self.running_var, 
            self.bn_weight, 
            self.bn_bias, 
            self.scaling_factor,
            self.kernel_size,
            self.padding
        )