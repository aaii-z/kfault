import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused sigmoid and sum reduction
cuda_code = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sigmoid_sum_kernel(const float* input, float* output, 
                                  int batch_size, int num_elements) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float local_sum = 0.0f;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        float val = input[bid * num_elements + i];
        local_sum += 1.0f / (1.0f + expf(-val));
    }

    shared[tid] = local_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[bid] = shared[0];
    }
}

torch::Tensor fused_sigmoid_sum_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int num_elements = input.numel() / batch_size;
    
    auto output = torch::zeros({batch_size}, input.options());
    
    int threads = 256;
    int blocks = batch_size;
    
    size_t shared_mem = threads * sizeof(float);
    
    sigmoid_sum_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_elements
    );
    
    return output;
}
"""

cpp_code = "torch::Tensor fused_sigmoid_sum_cuda(torch::Tensor input);"

# Load the custom CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_code,
    cuda_sources=cuda_code,
    functions=['fused_sigmoid_sum_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = fused_ops.fused_sigmoid_sum_cuda(x)
        return x