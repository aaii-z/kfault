import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operation
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operation_kernel(const float* x, const float* weight, float* output, 
                                       int batch_size, int input_size, int hidden_size, 
                                       float scaling_factor) {
    int i = blockIdx.x;
    if (i >= batch_size) return;

    extern __shared__ float x_shared[];
    // Load the i-th row of x into shared memory
    for (int k = threadIdx.x; k < input_size; k += blockDim.x) {
        x_shared[k] = x[i * input_size + k];
    }
    __syncthreads();

    float thread_sum = 0.0f;
    // Each thread processes a subset of weight rows
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        float dot_product = 0.0f;
        for (int k = 0; k < input_size; k++) {
            dot_product += x_shared[k] * weight[j * input_size + k];
        }
        thread_sum += dot_product / 2.0f;
    }

    // Reduce within the block
    __shared__ float block_shared[256];
    int tid = threadIdx.x;
    block_shared[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_shared[tid] += block_shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[i] = block_shared[0] * scaling_factor;
    }
}

torch::Tensor fused_operation_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = weight.size(0);
    auto output = torch::empty({batch_size, 1}, x.options());

    const int block_size = 256;
    const int shared_memory_size = input_size * sizeof(float);

    fused_operation_kernel<<<batch_size, block_size, shared_memory_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, input_size, hidden_size, scaling_factor
    );

    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_operation_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor);"

# Compile the inline CUDA code
fused_operation = load_inline(
    name="fused_operation",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_operation = fused_operation

    def forward(self, x):
        return self.fused_operation.fused_operation_cuda(x, self.weight, self.scaling_factor)