import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused softmax-subtract-swish-max operation
fused_softmax_swish_max_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

__global__ void fused_softmax_subtract_swish_max_kernel(
    const float* input,
    const float* subtract,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int spatial_size = depth * height * width;
    int total_elements = batch_size * spatial_size;
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    int b = index / spatial_size;
    int spatial_idx = index % spatial_size;
    int d = spatial_idx / (height * width);
    int hw_idx = spatial_idx % (height * width);
    int h = hw_idx / width;
    int w = hw_idx % width;

    const float* channel_data = input + b * channels * spatial_size + spatial_idx;
    float max_val = -FLT_MAX;
    
    // Find max value for softmax stability
    for (int c = 0; c < channels; ++c) {
        float val = channel_data[c * spatial_size];
        if (val > max_val) max_val = val;
    }

    // Compute softmax and apply operations
    float sum_exp = 0.0f;
    float swish_vals[16];  // Assuming max channels <= 16
    
    for (int c = 0; c < channels; ++c) {
        float val = channel_data[c * spatial_size];
        float exp_val = expf(val - max_val);
        sum_exp += exp_val;
        swish_vals[c] = exp_val;
    }

    float max_result = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
        // Softmax and subtract
        float softmax_val = swish_vals[c] / sum_exp;
        float subtracted = softmax_val - subtract[c];
        
        // Swish activation
        float swish_val = subtracted * (1.0f / (1.0f + expf(-subtracted)));
        
        // Track max value
        if (swish_val > max_result) max_result = swish_val;
    }

    output[index] = max_result;
}

torch::Tensor fused_softmax_subtract_swish_max_cuda(
    torch::Tensor input,
    torch::Tensor subtract
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    auto output = torch::empty({batch_size, depth, height, width}, input.options());
    
    int total_elements = batch_size * depth * height * width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_softmax_subtract_swish_max_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        subtract.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width
    );
    
    return output;
}
"""

fused_softmax_swish_max_cpp_source = """
torch::Tensor fused_softmax_subtract_swish_max_cuda(torch::Tensor input, torch::Tensor subtract);
"""

# Load the custom CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=fused_softmax_swish_max_cpp_source,
    cuda_sources=fused_softmax_swish_max_source,
    functions=['fused_softmax_subtract_swish_max_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.max_pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size, 
            stride=pool_stride, 
            padding=pool_padding
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = fused_ops.fused_softmax_subtract_swish_max_cuda(x, self.subtract)
        return x