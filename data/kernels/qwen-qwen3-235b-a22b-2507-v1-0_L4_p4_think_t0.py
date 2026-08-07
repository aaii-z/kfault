import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_mish_mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        const float threshold = 20.0f;
        
        // First Mish activation
        float y;
        if (x > threshold) {
            y = x;
        } else {
            float sp1 = log1pf(expf(x));
            y = x * tanhf(sp1);
        }
        
        // Second Mish activation
        float z;
        if (y > threshold) {
            z = y;
        } else {
            float sp2 = log1pf(expf(y));
            z = y * tanhf(sp2);
        }
        
        output[idx] = z;
    }
}

torch::Tensor fused_mish_mish_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    if (size == 0) {
        return output;
    }
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_mish_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    
    return output;
}
"""

fused_mish_mish_cpp = "torch::Tensor fused_mish_mish_cuda(torch::Tensor input);"

fused_mish_mish = load_inline(
    name="fused_mish_mish",
    cpp_sources=fused_mish_mish_cpp,
    cuda_sources=fused_mish_mish_source,
    functions=["fused_mish_mish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses two consecutive Mish activations into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_mish_mish = fused_mish_mish

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_mish_mish.fused_mish_mish_cuda(x)
        return x

batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]