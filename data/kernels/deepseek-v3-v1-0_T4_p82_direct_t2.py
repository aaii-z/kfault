import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for conv + tanh + scale + bias
fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_conv_activation_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float scaling_factor,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int output_height, int output_width) {
    
    const int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_c = blockIdx.z % out_channels;
    const int batch = blockIdx.z / out_channels;
    
    if (output_x >= output_width || output_y >= output_height || batch >= batch_size || out_c >= out_channels) {
        return;
    }
    
    float sum = 0.0f;
    
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int input_y = output_y + ky - kernel_size / 2;
                const int input_x = output_x + kx - kernel_size / 2;
                
                if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                    const int input_idx = ((batch * in_channels + in_c) * input_height + input_y) * input_width + input_x;
                    const int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    sum += bias[out_c];
    float result = tanhf(sum) * scaling_factor;
    
    const int output_idx = ((batch * out_channels + out_c) * output_height + output_y) * output_width + output_x;
    output[output_idx] = result;
}

torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, int kernel_size) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(0);
    
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
                              input.options().device(torch::kCUDA));
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (output_width + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        batch_size * out_channels
    );
    
    fused_conv_activation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_size, output_height, output_width
    );
    
    return output;
}
"""

fused_conv_activation_cpp_source = """
torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, int kernel_size);
"""

# Define the max pool kernel
max_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void max_pool_kernel(
    const float* input, float* output,
    int batch_size, int channels,
    int input_height, int input_width,
    int pool_size, int output_height, int output_width) {
    
    const int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z % channels;
    const int batch = blockIdx.z / channels;
    
    if (output_x >= output_width || output_y >= output_height || batch >= batch_size || channel >= channels) {
        return;
    }
    
    float max_val = -FLT_MAX;
    const int start_y = output_y * pool_size;
    const int start_x = output_x * pool_size;
    
    for (int py = 0; py < pool_size; ++py) {
        for (int px = 0; px < pool_size; ++px) {
            const int input_y = start_y + py;
            const int input_x = start_x + px;
            
            if (input_y < input_height && input_x < input_width) {
                const int input_idx = ((batch * channels + channel) * input_height + input_y) * input_width + input_x;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }
    
    const int output_idx = ((batch * channels + channel) * output_height + output_y) * output_width + output_x;
    output[output_idx] = max_val;
}

torch::Tensor max_pool_cuda(torch::Tensor input, int pool_size) {
    CHECK_INPUT(input);
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = input_height / pool_size;
    const int output_width = input_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, 
                              input.options().device(torch::kCUDA));
    
    dim3 block_size(16, 16);
    dim3 grid_size(
        (output_width + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        batch_size * channels
    );
    
    max_pool_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels,
        input_height, input_width,
        pool_size, output_height, output_width
    );
    
    return output;
}
"""

max_pool_cpp_source = "torch::Tensor max_pool_cuda(torch::Tensor input, int pool_size);"

# Compile the inline CUDA code
fused_conv_activation = load_inline(
    name="fused_conv_activation",
    cpp_sources=fused_conv_activation_cpp_source,
    cuda_sources=fused_conv_activation_source,
    functions=["fused_conv_activation_cuda"],
    verbose=False
)

max_pool_cuda = load_inline(
    name="max_pool_cuda",
    cpp_sources=max_pool_cpp_source,
    cuda_sources=max_pool_source,
    functions=["max_pool_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        self.pool_kernel_size = pool_kernel_size
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store CUDA operators
        self.fused_conv_activation = fused_conv_activation
        self.max_pool_cuda = max_pool_cuda

    def forward(self, x):
        # Fused convolution + tanh + scaling + bias
        x = self.fused_conv_activation.fused_conv_activation_cuda(
            x, self.weight, self.bias, self.scaling_factor, self.kernel_size
        )
        
        # Max pooling
        x = self.max_pool_cuda.max_pool_cuda(x, self.pool_kernel_size)
        
        return x