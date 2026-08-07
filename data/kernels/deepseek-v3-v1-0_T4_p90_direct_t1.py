import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Conv3D + LeakyReLU + Add + Clamp + GELU
fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
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
    int input_depth, int input_height, int input_width, int kernel_size) {
    
    const int output_depth = input_depth - kernel_size + 1;
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    
    const int output_size = batch_size * out_channels * output_depth * output_height * output_width;
    
    CUDA_KERNEL_LOOP(idx, output_size) {
        // Calculate output indices
        int n = idx / (out_channels * output_depth * output_height * output_width);
        int remainder = idx % (out_channels * output_depth * output_height * output_width);
        int c_out = remainder / (output_depth * output_height * output_width);
        remainder = remainder % (output_depth * output_height * output_width);
        int d_out = remainder / (output_height * output_width);
        remainder = remainder % (output_height * output_width);
        int h_out = remainder / output_width;
        int w_out = remainder % output_width;
        
        float result = bias[c_out];
        
        // Perform convolution
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kd = 0; kd < kernel_size; kd++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int d_in = d_out + kd;
                        int h_in = h_out + kh;
                        int w_in = w_out + kw;
                        
                        int input_idx = ((n * in_channels + c_in) * input_depth + d_in) * input_height * input_width + 
                                      h_in * input_width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kd) * kernel_size * kernel_size + 
                                       kh * kernel_size + kw;
                        
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Apply LeakyReLU
        result = leaky_relu(result, negative_slope);
        
        // Add sum tensor (broadcasted)
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
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    auto out_channels = weight.size(0);
    
    auto output_depth = input_depth - kernel_size + 1;
    auto output_height = input_height - kernel_size + 1;
    auto output_width = input_width - kernel_size + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                              input.options());
    
    int output_size = batch_size * out_channels * output_depth * output_height * output_width;
    int num_blocks = (output_size + kBlockSize - 1) / kBlockSize;
    
    fused_conv_activation_kernel<<<num_blocks, kBlockSize>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), sum_tensor.data_ptr<float>(),
        output.data_ptr<float>(), negative_slope, min_val, max_val,
        batch_size, in_channels, out_channels, input_depth, input_height, input_width, kernel_size);
    
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