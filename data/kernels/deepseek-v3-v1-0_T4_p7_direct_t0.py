import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Conv3D + Activations + Bias
fused_conv_activation_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ float leaky_relu(float x, float negative_slope) {
    return x > 0.0f ? x : x * negative_slope;
}

__device__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_activation_bias_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, 
    const int batch_size, const int in_channels, const int out_channels,
    const int depth, const int height, const int width,
    const int kernel_size, const int pad, const int stride,
    const float negative_slope) {
    
    const int output_depth = (depth - kernel_size) / stride + 1;
    const int output_height = (height - kernel_size) / stride + 1;
    const int output_width = (width - kernel_size) / stride + 1;
    
    const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    CUDA_KERNEL_LOOP(idx, total_elements) {
        const int n = idx / (out_channels * output_depth * output_height * output_width);
        const int oc = (idx / (output_depth * output_height * output_width)) % out_channels;
        const int od = (idx / (output_height * output_width)) % output_depth;
        const int oh = (idx / output_width) % output_height;
        const int ow = idx % output_width;
        
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        const int id = od * stride + kd;
                        const int ih = oh * stride + kh;
                        const int iw = ow * stride + kw;
                        
                        if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            const int input_idx = ((n * in_channels + ic) * depth + id) * height * width + ih * width + iw;
                            const int weight_idx = ((oc * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Apply ReLU
        sum = fmaxf(0.0f, sum);
        
        // Apply LeakyReLU
        sum = leaky_relu(sum, negative_slope);
        
        // Apply GELU
        sum = gelu_approx(sum);
        
        // Apply Sigmoid
        sum = 1.0f / (1.0f + expf(-sum));
        
        // Add bias
        sum += bias[oc];
        
        const int output_idx = ((n * out_channels + oc) * output_depth + od) * output_height * output_width + oh * output_width + ow;
        output[output_idx] = sum;
    }
}

torch::Tensor fused_conv_activation_bias_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int pad, float negative_slope) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    const int out_channels = weight.size(0);
    
    const int output_depth = (depth - kernel_size) / stride + 1;
    const int output_height = (height - kernel_size) / stride + 1;
    const int output_width = (width - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, 
                              torch::dtype(torch::kFloat32).device(input.device()));
    
    const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_activation_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, pad, stride,
        negative_slope);
    
    return output;
}
"""

fused_conv_activation_bias_cpp_source = """
torch::Tensor fused_conv_activation_bias_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int pad, float negative_slope);
"""

# Compile the inline CUDA code
fused_conv_activation_bias = load_inline(
    name="fused_conv_activation_bias",
    cpp_sources=fused_conv_activation_bias_cpp_source,
    cuda_sources=fused_conv_activation_bias_source,
    functions=["fused_conv_activation_bias_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.pad = 0
        self.negative_slope = 0.01
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Load custom CUDA operator
        self.fused_op = fused_conv_activation_bias

    def forward(self, x):
        return self.fused_op.fused_conv_activation_bias_cuda(
            x, self.weight, self.bias, 
            self.kernel_size, self.stride, self.pad, self.negative_slope
        )