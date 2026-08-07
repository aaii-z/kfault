import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused convolution, scaling, and min reduction
fused_conv_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_conv_scale_min_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const float scale_factor,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int padding,
    const int stride
) {
    const int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    const int n = blockIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (n >= batch_size || h >= output_h || w >= output_w) {
        return;
    }
    
    float min_val = FLT_MAX;
    
    for (int oc = 0; oc < out_channels; oc++) {
        float result = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int input_h = h * stride - padding + kh;
                    const int input_w = w * stride - padding + kw;
                    
                    if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                        const int input_idx = ((n * in_channels + ic) * height + input_h) * width + input_w;
                        const int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            result += bias[oc];
        }
        
        result *= scale_factor;
        
        if (result < min_val) {
            min_val = result;
        }
    }
    
    const int output_idx = (n * output_h + h) * output_w + w;
    output[output_idx] = min_val;
}

torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    int padding,
    int stride
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int output_h = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, 1, output_h, output_w}, input.options());
    
    const dim3 block_size(1, 16, 16);
    const dim3 grid_size(
        batch_size,
        (output_h + block_size.y - 1) / block_size.y,
        (output_w + block_size.z - 1) / block_size.z
    );
    
    fused_conv_scale_min_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        scale_factor,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        padding,
        stride
    );
    
    return output;
}
"""

fused_conv_scale_min_cpp_source = """
torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    int padding,
    int stride
);
"""

# Compile the inline CUDA code
fused_conv_scale_min = load_inline(
    name="fused_conv_scale_min",
    cpp_sources=fused_conv_scale_min_cpp_source,
    cuda_sources=fused_conv_scale_min_source,
    functions=["fused_conv_scale_min_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.scale_factor = scale_factor
        
        # Initialize convolution parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.padding = 0  # No padding to match original output size
        self.stride = 1
        
        self.fused_op = fused_conv_scale_min

    def forward(self, x):
        return self.fused_op.fused_conv_scale_min_cuda(
            x, self.weight, self.bias, self.scale_factor, self.padding, self.stride
        )