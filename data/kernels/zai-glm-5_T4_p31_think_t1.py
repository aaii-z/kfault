import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min, add, and scale
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_ops_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                 const float* __restrict__ bias, float constant_val, float scale_val, 
                                 int total_elements, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process 4 elements at a time using float4 for memory coalescing
    int limit = total_elements / 4;
    
    for (int i = idx; i < limit; i += stride) {
        float4 val = reinterpret_cast<const float4*>(input)[i];
        
        // Calculate the channel index for the first element in the chunk
        // Since HW is divisible by 4 (128*128), a float4 chunk always lies within the same channel
        int base_idx = i * 4;
        int c = (base_idx / HW) % C;
        float b = bias[c];
        
        // Apply operations: min(constant) -> add(bias) -> mul(scale)
        val.x = (min(val.x, constant_val) + b) * scale_val;
        val.y = (min(val.y, constant_val) + b) * scale_val;
        val.z = (min(val.z, constant_val) + b) * scale_val;
        val.w = (min(val.w, constant_val) + b) * scale_val;
        
        reinterpret_cast<float4*>(output)[i] = val;
    }

    // Handle remaining elements if total_elements is not divisible by 4
    for (int i = limit * 4 + idx; i < total_elements; i += stride) {
        int c = (i / HW) % C;
        float b = bias[c];
        output[i] = (min(input[i], constant_val) + b) * scale_val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float constant_val, float scale_val) {
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    
    // Get dimensions
    // Assuming NCHW format
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int HW = H * W;

    const int block_size = 256;
    // Calculate grid size to cover all elements (vectorized)
    int num_blocks = (total_elements / 4 + block_size - 1) / block_size;
    
    fused_ops_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        constant_val, 
        scale_val, 
        total_elements, 
        C, 
        HW
    );
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float constant_val, float scale_val);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel to fuse min, add, and mul operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # Apply the fused kernel: min(constant) + bias * scaling_factor
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.constant_value, self.scaling_factor)
        return x