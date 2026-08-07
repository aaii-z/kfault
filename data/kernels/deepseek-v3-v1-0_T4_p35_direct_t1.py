import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel for conv, subtract, hardswish, maxpool, and mish
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float hardswish_activation(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__device__ __forceinline__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_conv_subtract_activation_pool_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size, int output_height, 
    int output_width, float subtract_value, int pool_kernel_size, int stride) {
    
    const int output_h = blockIdx.y * blockDim.y + threadIdx.y;
    const int output_w = blockIdx.z * blockDim.z + threadIdx.z;
    const int out_c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.w;
    
    if (out_c >= out_channels || output_h >= output_height || output_w >= output_width || b >= batch_size) {
        return;
    }
    
    // Convolution operation
    float conv_result = 0.0f;
    if (bias != nullptr) {
        conv_result = bias[out_c];
    }
    
    const int pad = kernel_size / 2;
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int input_h = output_h * stride + kh - pad;
                const int input_w = output_w * stride + kw - pad;
                
                if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                    const int input_idx = ((b * in_channels + in_c) * input_height + input_h) * input_width + input_w;
                    const int weight_idx = ((out_c * in_channels + in_c) * kernel_size + kh) * kernel_size + kw;
                    conv_result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Subtract value
    conv_result -= subtract_value;
    
    // HardSwish activation
    conv_result = hardswish_activation(conv_result);
    
    // MaxPool operation
    float max_val = -FLT_MAX;
    for (int ph = 0; ph < pool_kernel_size; ++ph) {
        for (int pw = 0; pw < pool_kernel_size; ++pw) {
            const int pool_h = output_h * pool_kernel_size + ph;
            const int pool_w = output_w * pool_kernel_size + pw;
            
            if (pool_h < input_height && pool_w < input_width) {
                max_val = fmaxf(max_val, conv_result);
            }
        }
    }
    
    // Mish activation
    max_val = mish_activation(max_val);
    
    const int output_idx = ((b * out_channels + out_c) * output_height + output_h) * output_width + output_w;
    output[output_idx] = max_val;
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
    float subtract_value, int pool_kernel_size) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) {
        CHECK_INPUT(bias);
    }
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int stride = 1;
    
    const int output_height = (input_height - pool_kernel_size) / pool_kernel_size + 1;
    const int output_width = (input_width - pool_kernel_size) / pool_kernel_size + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
                              input.options());
    
    const dim3 block_size(8, 4, 4);
    const dim3 grid_size(
        (out_channels + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        (output_width + block_size.z - 1) / block_size.z,
        batch_size
    );
    
    fused_conv_subtract_activation_pool_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size, output_height, output_width,
        subtract_value, pool_kernel_size, stride);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float subtract_value, int pool_kernel_size);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        
        # Initialize weights and bias like nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.subtract_value, self.pool_kernel_size
        )