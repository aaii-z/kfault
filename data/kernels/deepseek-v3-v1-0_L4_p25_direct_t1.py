import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution + min + tanh operations
fused_conv_min_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int padding, int stride,
    int output_h, int output_w) {
    
    // Calculate output position
    const int n = blockIdx.x;  // batch index
    const int h = blockIdx.y;  // output height index
    const int w = blockIdx.z;  // output width index
    
    if (n >= batch_size || h >= output_h || w >= output_w) return;
    
    // Initialize min value for this spatial position
    float min_val = FLT_MAX;
    
    // For each output channel, compute convolution and find min
    for (int oc = 0; oc < out_channels; oc++) {
        float conv_result = 0.0f;
        
        // Convolution operation
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int input_h = h * stride + kh - padding;
                    int input_w = w * stride + kw - padding;
                    
                    if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                        int input_idx = n * (in_channels * height * width) + 
                                      ic * (height * width) + 
                                      input_h * width + input_w;
                        
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
    
    // Write output (only one channel since we reduced)
    int output_idx = n * (output_h * output_w) + h * output_w + w;
    output[output_idx] = tanh2;
}

torch::Tensor fused_conv_min_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int padding, int stride) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    const int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, 1, output_h, output_w}, input.options());
    
    dim3 grid_size(batch_size, output_h, output_w);
    dim3 block_size(1);
    
    fused_conv_min_tanh_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, padding, stride, output_h, output_w);
    
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