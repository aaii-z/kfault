import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused convolution + GELU + average pooling CUDA kernel
fused_conv_gelu_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/M_PI) * (x + 0.044715 * x*x*x)))
__device__ float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void fused_conv_gelu_pool_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, 
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size,
    int stride, int padding) {
    
    int b = blockIdx.x;
    int out_c = blockIdx.y;
    int out_h = threadIdx.x;
    int out_w = threadIdx.y;
    
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    if (out_h >= output_height || out_w >= output_width) return;
    
    float conv_result = 0.0f;
    
    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int input_h = out_h * stride + kh - padding;
                int input_w = out_w * stride + kw - padding;
                
                if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                    int input_idx = ((b * in_channels + in_c) * input_height + input_h) * input_width + input_w;
                    int weight_idx = ((out_c * in_channels + in_c) * kernel_size + kh) * kernel_size + kw;
                    conv_result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    if (bias != nullptr) {
        conv_result += bias[out_c];
    }
    
    // Apply GELU activation
    float gelu_result = gelu_approx(conv_result);
    
    // Store intermediate result
    __shared__ float block_sum[1];
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        block_sum[0] = 0.0f;
    }
    __syncthreads();
    
    atomicAdd(&block_sum[0], gelu_result);
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float avg = block_sum[0] / (output_height * output_width);
        int final_idx = b * out_channels + out_c;
        output[final_idx] = avg;
    }
}

torch::Tensor fused_conv_gelu_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    // Create final output tensor
    auto final_output = torch::empty({batch_size, out_channels}, input.options());
    
    // Configure kernel launch parameters
    dim3 grid_size(batch_size, out_channels);
    dim3 block_size(output_height, output_width);
    
    int shared_mem_size = sizeof(float);
    
    fused_conv_gelu_pool_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        final_output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size,
        stride, padding);
    
    return final_output;
}
"""

fused_conv_gelu_pool_cpp_source = """
torch::Tensor fused_conv_gelu_pool_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code
fused_conv_gelu_pool = load_inline(
    name="fused_conv_gelu_pool",
    cpp_sources=fused_conv_gelu_pool_cpp_source,
    cuda_sources=fused_conv_gelu_pool_source,
    functions=["fused_conv_gelu_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_op = fused_conv_gelu_pool

    def forward(self, x):
        return self.fused_op.fused_conv_gelu_pool_cuda(x, self.weight, self.bias, self.kernel_size, self.stride, self.padding)