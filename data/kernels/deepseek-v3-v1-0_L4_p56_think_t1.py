import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sigmoid + sum
sigmoid_sum_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sigmoid_sum_kernel(const float* input, float* output, int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int index = row * cols;

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = input[index + i];
        sum += 1.0f / (1.0f + expf(-val));
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = sdata[0];
    }
}

torch::Tensor sigmoid_sum_cuda(torch::Tensor input) {
    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::zeros({rows, 1}, input.options());

    int threads = 256;
    size_t shared_mem_size = threads * sizeof(float);

    sigmoid_sum_kernel<<<rows, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    return output;
}
"""

sigmoid_sum_cpp_source = """
torch::Tensor sigmoid_sum_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
sigmoid_sum = load_inline(
    name="sigmoid_sum",
    cpp_sources=sigmoid_sum_cpp_source,
    cuda_sources=sigmoid_sum_kernel_source,
    functions=["sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.linear(x)
        x = sigmoid_sum.sigmoid_sum_cuda(x)
        return x