import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction
min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void min_reduction_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int n = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = FLT_MAX;

    for (int c = tid; c < C; c += blockDim.x) {
        int idx = n * (C * H * W) + c * (H * W) + h * W + w;
        if (input[idx] < sdata[tid]) {
            sdata[tid] = input[idx];
        }
    }
    __syncthreads();

    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            if (sdata[tid] > sdata[tid+s]) {
                sdata[tid] = sdata[tid+s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[n * (H * W) + h * W + w] = sdata[0];
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor input) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    auto output = torch::empty({N, 1, H, W}, input.options());
    
    dim3 grid(N, H, W);
    int block_size = 256;
    int shared_mem_size = block_size * sizeof(float);
    
    min_reduction_kernel<<<grid, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    return output;
}
"""

min_reduction_cpp_source = "torch::Tensor min_reduction_cuda(torch::Tensor input);"

# Compile the inline CUDA code
min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        
        # Scale the convolution weights and bias
        with torch.no_grad():
            self.conv.weight *= scale_factor
            if self.conv.bias is not None:
                self.conv.bias *= scale_factor
                
        self.min_reduction = min_reduction

    def forward(self, x):
        x = self.conv(x)
        x = self.min_reduction.min_reduction_cuda(x)
        return x