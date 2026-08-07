import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Conv3D + LeakyReLU + Add + Clamp + GELU
fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

constexpr int kBlockSize = 256;

__device__ float leaky_relu(float x, float negative_slope) {
    return x > 0 ? x : x * negative_slope;
}

__device__ float gelu(float x) {
    return 0.5 * x * (1.0 + tanhf(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

__global__ void fused_conv_activation_kernel(
    const float* input, const float* weight, const float* bias, const float* sum_tensor,
    float* output, float negative_slope, float min_val, float max_val,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width, int kernel_size) {
    
    const int output_size = batch_size * out_channels * (depth - kernel_size + 1) * 
                           (height - kernel_size + 1) * (width - kernel_size + 1);
    
    CUDA_KERNEL_LOOP(idx, output_size) {
        // Calculate output indices
        int n = idx / (out_channels * (depth - kernel_size + 1) * (height - kernel_size + 1) * (width - kernel_size + 1));
        int remainder = idx % (out_channels * (depth - kernel_size + 1) * (height - kernel_size + 1) * (width - kernel_size + 1));
        int c_out = remainder / ((depth - kernel_size + 1) * (height - kernel_size + 1) * (width - kernel_size + 1));
        remainder = remainder % ((depth - kernel_size + 1) * (height - kernel_size + 1) * (width - kernel_size + 1));
        int d_out = remainder / ((height - kernel_size + 1) * (width - kernel_size + 1));
        remainder = remainder % ((height - kernel_size + 1) * (width - kernel_size + 1));
        int h_out = remainder / (width - kernel_size + 1);
        int w_out = remainder % (width - kernel_size + 1);
        
        float result = bias[c_out];
        
        // Perform convolution
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kd = 0; kd < kernel_size; kd++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int d_in = d_out + kd;
                        int h_in = h_out + kh;
                        int w_in = w_out + kw;
                        
                        int input_idx = ((n * in_channels + c_in) * depth + d_in) * height * width + 
                                      h_in * width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kd) * kernel_size * kernel_size + 
                                       kh * kernel_size + kw;
                        
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Apply LeakyReLU
        result = leaky_relu(result, negative_slope);
        
        // Add sum tensor
        result += sum_tensor[c_out];
        
        // Clamp
        result = fminf(fmaxf(result, min_val), max_val);
        
        // Apply GELU
        result = gelu(result);
        
        output[idx] = result;
    }
}

torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor sum_tensor,
    float negative_slope, float min_val, float max_val, int kernel_size) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out_channels = weight.size(0);
    
    auto output_depth = depth - kernel_size + 1;
    auto output_height = height - kernel_size + 1;
    auto output_width = width - kernel_size + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                              input.options());
    
    int output_size = batch_size * out_channels * output_depth * output_height * output_width;
    int num_blocks = (output_size + kBlockSize - 1) / kBlockSize;
    
    fused_conv_activation_kernel<<<num_blocks, kBlockSize>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), sum_tensor.data_ptr<float>(),
        output.data_ptr<float>(), negative_slope, min_val, max_val,
        batch_size, in_channels, out_channels, depth, height, width, kernel_size);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_activation_cpp_source = """
torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor sum_tensor,
    float negative_slope, float min_val, float max_val, int kernel_size);
"""

# Compile the inline CUDA code
fused_conv_activation = load_inline(
    name="fused_conv_activation",
    cpp_sources=fused_conv_activation_cpp_source,
    cuda_sources=fused_conv_activation_source,
    functions=["fused_conv_activation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize weights and bias for convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        
        # Store the custom CUDA operator
        self.fused_conv_activation = fused_conv_activation

    def forward(self, x):
        return self.fused_conv_activation.fused_conv_activation_cuda(
            x, self.weight, self.bias, self.sum_tensor, 0.2, -1.0, 1.0, self.kernel_size
        )