import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for ConvTranspose3d + LeakyReLU + Elementwise Multiply + LeakyReLU + MaxPool3d
fused_operator_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int kMaxThreadsPerBlock = 256;

__device__ __forceinline__ float leaky_relu(float x, float negative_slope) {
    return x > 0.0f ? x : x * negative_slope;
}

__global__ void fused_conv_transpose_leaky_relu_multiply_leaky_relu_maxpool_kernel(
    const float* input, const float* weight, const float* bias, const float* multiplier,
    float* output, float negative_slope,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride, int padding, int output_padding) {
    
    // Calculate output dimensions for transposed convolution
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_depth + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_height + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_width + output_padding;
    
    // Calculate pooled dimensions
    int pooled_depth = (output_depth + 1) / 2;
    int pooled_height = (output_height + 1) / 2;
    int pooled_width = (output_width + 1) / 2;
    
    const int total_output_elements = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    const int n = total_output_elements;
    
    CUDA_KERNEL_LOOP(index, n) {
        // Calculate output indices
        const int pw = index % pooled_width;
        const int ph = (index / pooled_width) % pooled_height;
        const int pd = (index / (pooled_width * pooled_height)) % pooled_depth;
        const int c_out = (index / (pooled_width * pooled_height * pooled_depth)) % out_channels;
        const int b = index / (pooled_width * pooled_height * pooled_depth * out_channels);
        
        // Calculate input window for max pooling
        const int d_start = pd * 2;
        const int d_end = min(d_start + 2, output_depth);
        const int h_start = ph * 2;
        const int h_end = min(h_start + 2, output_height);
        const int w_start = pw * 2;
        const int w_end = min(w_start + 2, output_width);
        
        float max_val = -FLT_MAX;
        
        // Iterate over pooling window
        for (int d = d_start; d < d_end; ++d) {
            for (int h = h_start; h < h_end; ++h) {
                for (int w = w_start; w < w_end; ++w) {
                    // Perform transposed convolution
                    float conv_val = bias[c_out];
                    
                    // Iterate over input channels
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        // Iterate over kernel
                        for (int kd = 0; kd < kernel_depth; ++kd) {
                            for (int kh = 0; kh < kernel_height; ++kh) {
                                for (int kw = 0; kw < kernel_width; ++kw) {
                                    int input_d = d - kd + padding;
                                    int input_h = h - kh + padding;
                                    int input_w = w - kw + padding;
                                    
                                    if (input_d >= 0 && input_d < input_depth && 
                                        input_h >= 0 && input_h < input_height && 
                                        input_w >= 0 && input_w < input_width) {
                                        int input_idx = ((b * in_channels + c_in) * input_depth + input_d) * input_height * input_width + 
                                                       input_h * input_width + input_w;
                                        int weight_idx = ((c_out * in_channels + c_in) * kernel_depth + kd) * kernel_height * kernel_width + 
                                                        kh * kernel_width + kw;
                                        conv_val += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    
                    // Apply first LeakyReLU
                    float leaky1 = leaky_relu(conv_val, negative_slope);
                    
                    // Multiply by learnable parameter
                    float multiplied = leaky1 * multiplier[c_out];
                    
                    // Apply second LeakyReLU
                    float leaky2 = leaky_relu(multiplied, negative_slope);
                    
                    // Update max value
                    if (leaky2 > max_val) {
                        max_val = leaky2;
                    }
                }
            }
        }
        
        // Store the max pooled value
        output[index] = max_val;
    }
}

torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor multiplier,
    float negative_slope, int stride, int padding, int output_padding) {
    
    // Get input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);
    
    // Calculate pooled dimensions
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_depth + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_height + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_width + output_padding;
    
    int pooled_depth = (output_depth + 1) / 2;
    int pooled_height = (output_height + 1) / 2;
    int pooled_width = (output_width + 1) / 2;
    
    // Create output tensor
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({batch_size, out_channels, pooled_depth, pooled_height, pooled_width}, options);
    
    // Calculate grid and block dimensions
    const int total_elements = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    const int threads_per_block = kMaxThreadsPerBlock;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    fused_conv_transpose_leaky_relu_multiply_leaky_relu_maxpool_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), multiplier.data_ptr<float>(),
        output.data_ptr<float>(), negative_slope,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        stride, padding, output_padding);
    
    return output;
}
"""

fused_operator_cpp_source = """
torch::Tensor fused_operator_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor multiplier,
    float negative_slope, int stride, int padding, int output_padding);
"""

# Compile the inline CUDA code
fused_operator = load_inline(
    name="fused_operator",
    cpp_sources=fused_operator_cpp_source,
    cuda_sources=fused_operator_source,
    functions=["fused_operator_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.negative_slope = 0.2
        
        # Initialize weights and bias for transposed convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        
        # Store the fused operator
        self.fused_operator = fused_operator

    def forward(self, x):
        return self.fused_operator.fused_operator_cuda(
            x, self.weight, self.bias, self.multiplier,
            self.negative_slope,
            self.stride, self.padding, self.output_padding
        )