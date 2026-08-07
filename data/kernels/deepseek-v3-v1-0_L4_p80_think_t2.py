import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void max_reduce_kernel(const float* input, float* output, int rows, int cols, int reduce_dim) {
    extern __shared__ float shared[];

    if (reduce_dim == 1) {
        int row = blockIdx.x;
        int col = threadIdx.x;
        float max_val = -INFINITY;
        for (int i = col; i < cols; i += blockDim.x) {
            float val = input[row * cols + i];
            if (val > max_val) max_val = val;
        }
        shared[threadIdx.x] = max_val;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (shared[threadIdx.x + s] > shared[threadIdx.x]) {
                    shared[threadIdx.x] = shared[threadIdx.x + s];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            output[row] = shared[0];
        }
    } else {
        int col = blockIdx.x;
        int row = threadIdx.x;
        float max_val = -INFINITY;
        for (int i = row; i < rows; i += blockDim.x) {
            float val = input[i * cols + col];
            if (val > max_val) max_val = val;
        }
        shared[threadIdx.x] = max_val;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (shared[threadIdx.x + s] > shared[threadIdx.x]) {
                    shared[threadIdx.x] = shared[threadIdx.x + s];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            output[col] = shared[0];
        }
    }
}

__global__ void subtract_mean_and_gelu_kernel(const float* input, float* output, int rows, int cols, const float* max_vals, const float* mean_vals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        float val = input[row * cols + col] - max_vals[row] - mean_vals[row];
        output[row * cols + col] = gelu(val);
    }
}

torch::Tensor fused_max_sub_gelu_cuda(torch::Tensor input, int reduce_dim) {
    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::empty_like(input);

    auto max_vals = torch::empty({rows}, input.options());
    auto mean_vals = torch::empty({rows}, input.options());

    if (reduce_dim == 1) {
        max_reduce_kernel<<<rows, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            input.data_ptr<float>(), max_vals.data_ptr<float>(), rows, cols, reduce_dim
        );

        dim3 blocks((rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE);
        subtract_mean_and_gelu_kernel<<<blocks, BLOCK_SIZE>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), rows, cols, 
            max_vals.data_ptr<float>(), mean_vals.data_ptr<float>()
        );
    } else {
        auto max_vals_col = torch::empty({cols}, input.options());
        max_reduce_kernel<<<cols, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            input.data_ptr<float>(), max_vals_col.data_ptr<float>(), rows, cols, reduce_dim
        );
        
        dim3 blocks((rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE);
        subtract_mean_and_gelu_kernel<<<blocks, BLOCK_SIZE>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), rows, cols,
            max_vals_col.data_ptr<float>(), mean_vals.data_ptr<float>()
        );
    }

    return output;
}
"""

custom_op_cpp_source = "torch::Tensor fused_max_sub_gelu_cuda(torch::Tensor input, int reduce_dim);"

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=custom_op_cpp_source,
    cuda_sources=custom_op_source,
    functions=["fused_max_sub_gelu_cuda"],
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
        x = fused_ops.fused_max_sub_gelu_cuda(x, self.max_dim)
        return x