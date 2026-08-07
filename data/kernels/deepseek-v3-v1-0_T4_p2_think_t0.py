import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias add and double clamp with scaling
fused_bias_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_clamp_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float scaling_factor,
    float min_val,
    float max_val
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (index < total_elements) {
        int c = (index / (height * width)) % channels;
        float bias_val = bias[c];
        float val = input[index] + bias_val;
        
        // First clamp
        val = fminf(fmaxf(val, min_val), max_val);
        // Scale
        val = val * scaling_factor;
        // Second clamp
        val = fminf(fmaxf(val, min_val), max_val);
        // Descaling
        val = val / scaling_factor;
        
        output[index] = val;
    }
}

torch::Tensor fused_bias_clamp_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::empty_like(input);
    
    int total_elements = batch_size * channels * height * width;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_bias_clamp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        scaling_factor,
        0.0f,  // min_val
        1.0f   // max_val
    );
    
    return output;
}
"""

fused_bias_clamp_cpp_source = "torch::Tensor fused_bias_clamp_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"

# Compile the inline CUDA code
fused_bias_clamp = load_inline(
    name="fused_bias_clamp",
    cpp_sources=fused_bias_clamp_cpp_source,
    cuda_sources=fused_bias_clamp_source,
    functions=["fused_bias_clamp_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_bias_clamp = fused_bias_clamp

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_bias_clamp.fused_bias_clamp_cuda(x, self.bias, self.scaling_factor)
        return x