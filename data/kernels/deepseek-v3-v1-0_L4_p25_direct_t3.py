import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution + min + tanh operations
fused_conv_min_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output, 
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_size, int padding, int stride,
    int output_h, int output_w) {
    
    const int n = blockIdx.z;  // batch index
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;  // output height
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;  // output width
    
    if (n >= batch_size || oh >= output_h || ow >= output_w) return;
    
    float min_val = INFINITY;
    
    // For each output channel
    for (int oc = 0; oc < out_channels; oc++) {
        float conv_result = 0.0f;
        
        // For each input channel
        for (int ic = 0; ic < in_channels; ic++) {
            // For each kernel element
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    
                    if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                        int input_idx = n * (in_channels * input_h * input_w) + 
                                     ic * (input_h * input_w) + 
                                     ih * input_w + iw;
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size) + 
                                       ic * (kernel_size * kernel_size) + 
                                       kh * kernel_size + kw;
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        if (bias != nullptr) {
            conv_result += bias[oc];
        }
        
        // Update min value
        min_val = fminf(min_val, conv_result);
    }
    
    // Apply double tanh
    float tanh1 = tanhf(min_val);
    float tanh2 = tanhf(tanh1);
    
    // Write output
    int output_idx = n * (output_h * output_w) + oh * output_w + ow;
    output[output_idx] = tanh2;
}

torch::Tensor fused_conv_min_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int padding, int stride) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_h = input.size(2);
    auto input_w = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    const int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, 1, output_h, output_w}, input.options());
    
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim(
        (output_w + block_dim.x - 1) / block_dim.x,
        (output_h + block_dim.y - 1) / block_dim.y,
        batch_size
    );
    
    fused_conv_min_tanh_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        input_h, input_w, kernel_size, padding, stride, output_h, output_w);
    
    cudaDeviceSynchronize();
    return output;
}
"""

fused_conv_min_tanh_cpp_source = """
torch::Tensor fused_conv_min_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int padding, int stride);
"""

# Compile the inline CUDA code
fused_conv_min_tanh = load_inline(
    name="fused_conv_min_tanh",
    cpp_sources=fused_conv_min_tanh_cpp_source,
    cuda_sources=fused_conv_min_tanh_source,
    functions=["fused_conv_min_tanh_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs fused convolution + min + tanh operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = 1
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        self.fused_op = fused_conv_min_tanh

    def forward(self, x):
        return self.fused_op.fused_conv_min_tanh_cuda(
            x, self.weight, self.bias, self.padding, self.stride
        )