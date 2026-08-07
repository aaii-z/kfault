import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution, subtraction, tanh, subtraction, and average pooling
fused_conv_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_sub_tanh_sub_avgpool_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size, int output_height, int output_width,
    float subtract1_value, float subtract2_value, int pool_kernel_size, int pool_stride) {
    
    const int pooled_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int pooled_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_c = blockIdx.z % out_channels;
    const int batch = blockIdx.z / out_channels;
    
    if (pooled_x >= output_width || pooled_y >= output_height || out_c >= out_channels || batch >= batch_size) {
        return;
    }
    
    const int conv_output_height = input_height - kernel_size + 1;
    const int conv_output_width = input_width - kernel_size + 1;
    
    // Average pooling region
    const int start_y = pooled_y * pool_stride;
    const int start_x = pooled_x * pool_stride;
    const int end_y = min(start_y + pool_kernel_size, conv_output_height);
    const int end_x = min(start_x + pool_kernel_size, conv_output_width);
    
    float pool_sum = 0.0f;
    int pool_count = 0;
    
    // Iterate over pooling region
    for (int y = start_y; y < end_y; ++y) {
        for (int x = start_x; x < end_x; ++x) {
            // Convolution operation for this position
            float conv_result = 0.0f;
            
            for (int in_c = 0; in_c < in_channels; ++in_c) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int input_y = y + ky;
                        int input_x = x + kx;
                        
                        if (input_y < input_height && input_x < input_width) {
                            int input_idx = ((batch * in_channels + in_c) * input_height + input_y) * input_width + input_x;
                            int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                            
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            // Add bias if defined
            if (bias != nullptr) {
                conv_result += bias[out_c];
            }
            
            // First subtraction
            conv_result -= subtract1_value;
            
            // Tanh activation
            float tanh_result = tanhf(conv_result);
            
            // Second subtraction
            tanh_result -= subtract2_value;
            
            pool_sum += tanh_result;
            pool_count++;
        }
    }
    
    float pool_result = pool_count > 0 ? pool_sum / pool_count : 0.0f;
    
    int output_idx = ((batch * out_channels + out_c) * output_height + pooled_y) * output_width + pooled_x;
    output[output_idx] = pool_result;
}

torch::Tensor fused_conv_sub_tanh_sub_avgpool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float subtract1_value, float subtract2_value, int pool_kernel_size) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int conv_output_height = input_height - kernel_size + 1;
    const int conv_output_width = input_width - kernel_size + 1;
    const int pool_stride = pool_kernel_size;
    const int pooled_height = (conv_output_height + pool_stride - 1) / pool_stride;
    const int pooled_width = (conv_output_width + pool_stride - 1) / pool_stride;
    
    auto output = torch::zeros({batch_size, out_channels, pooled_height, pooled_width}, input.options());
    
    const dim3 block_size(8, 8);
    const dim3 grid_size(
        (pooled_width + block_size.x - 1) / block_size.x,
        (pooled_height + block_size.y - 1) / block_size.y,
        batch_size * out_channels
    );
    
    fused_conv_sub_tanh_sub_avgpool_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size, pooled_height, pooled_width,
        subtract1_value, subtract2_value, pool_kernel_size, pool_stride
    );
    
    return output;
}
"""

fused_conv_ops_cpp_source = "torch::Tensor fused_conv_sub_tanh_sub_avgpool_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float subtract1_value, float subtract2_value, int pool_kernel_size);"

# Compile the inline CUDA code
fused_conv_ops = load_inline(
    name="fused_conv_ops",
    cpp_sources=fused_conv_ops_cpp_source,
    cuda_sources=fused_conv_ops_source,
    functions=["fused_conv_sub_tanh_sub_avgpool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        
        # Initialize weights and bias for convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_ops = fused_conv_ops

    def forward(self, x):
        return self.fused_ops.fused_conv_sub_tanh_sub_avgpool_cuda(
            x, self.weight, self.bias, 
            self.subtract1_value, self.subtract2_value, 
            self.kernel_size_pool
        )