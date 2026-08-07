import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void max_reduction_dim1_kernel(const float* input, float* output, int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = sdata[0];
    }
}

__global__ void max_reduction_dim0_kernel(const float* input, float* output, int rows, int cols) {
    extern __shared__ float sdata[];
    int col = blockIdx.x;
    int tid = threadIdx.x;

    float max_val = -INFINITY;
    for (int i = tid; i < rows; i += blockDim.x) {
        float val = input[i * cols + col];
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[col] = sdata[0];
    }
}

torch::Tensor custom_max_reduction_cuda(torch::Tensor input, int dim) {
    auto sizes = input.sizes();
    int rows = sizes[0];
    int cols = sizes[1];

    if (dim == 1) {
        auto output = torch::empty({rows, 1}, input.options());
        int block_size = 256;
        max_reduction_dim1_kernel<<<rows, block_size, block_size * sizeof(float)>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), rows, cols
        );
        return output;
    } else if (dim == 0) {
        auto output = torch::empty({1, cols}, input.options());
        int block_size = 256;
        max_reduction_dim0_kernel<<<cols, block_size, block_size * sizeof(float)>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), rows, cols
        );
        return output;
    } else {
        return torch::Tensor();
    }
}
"""

max_reduction_cpp_source = "torch::Tensor custom_max_reduction_cuda(torch::Tensor input, int dim);"

max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["custom_max_reduction_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        x = self.gemm(x)
        x = max_reduction.custom_max_reduction_cuda(x, self.max_dim)
        x = x - x.mean(dim=1, keepdim=True)
        x = torch.nn.functional.gelu(x)
        return x