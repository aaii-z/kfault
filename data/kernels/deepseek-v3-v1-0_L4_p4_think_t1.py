import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for double Mish activation
double_mish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ float mish_activation(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void double_mish_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        val = mish_activation(val);
        output[idx] = mish_activation(val);
    }
}

torch::Tensor double_mish_cuda(torch::Tensor input) {
    CHECK_INPUT(input);
    
    auto output = torch::empty_like(input);
    int num_elements = output.numel();
    
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    double_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements
    );
    
    return output;
}
"""

double_mish_cpp_source = "torch::Tensor double_mish_cuda(torch::Tensor input);"

# Compile the inline CUDA code
double_mish = load_inline(
    name="double_mish",
    cpp_sources=double_mish_cpp_source,
    cuda_sources=double_mish_source,
    functions=["double_mish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.double_mish = double_mish

    def forward(self, x):
        x = self.conv(x)
        x = self.double_mish.double_mish_cuda(x)
        return x