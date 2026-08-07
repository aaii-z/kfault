import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_ops_kernel(
    const float* input, 
    const float* bias, 
    const float* original_x, 
    float* output, 
    int total_elements,
    int bias_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // Calculate spatial and channel indices
        int spatial_idx = idx % (depth * height * width);
        int channel_idx = (idx / (depth * height * width)) % channels;
        int bias_idx = channel_idx % bias_size;
        
        float input_val = input[idx];
        float bias_val = bias[bias_idx];
        float orig_val = original_x[idx];
        
        // Fused operations: (input + bias + orig) * orig + orig
        // This is equivalent to: x = x + bias; x = x + orig_x; x = x * orig_x; x = x + orig_x;
        // Optimized: ((input + bias + orig) * orig) + orig
        float result = ((input_val + bias_val + orig_val) * orig_val) + orig_val;
        output[idx] = result;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor original_x) {
    int total_elements = input.numel();
    auto output = torch::zeros_like(input);
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int bias_size = bias.numel();
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        original_x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        bias_size,
        channels,
        depth,
        height,
        width
    );
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor original_x);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        original_x = x.clone().detach()
        x = self.fused_ops.fused_ops_cuda(x, self.bias, original_x)
        return x