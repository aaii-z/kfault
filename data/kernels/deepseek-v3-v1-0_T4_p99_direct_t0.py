import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GELU + Softmax CUDA kernel
fused_gelu_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_gelu_softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int col = threadIdx.x;
    
    if (row >= rows) return;
    
    extern __shared__ float shared_mem[];
    float* row_max = shared_mem;
    float* row_sum = shared_mem + blockDim.y;
    
    // Find maximum value in the row
    float max_val = -INFINITY;
    for (int i = col; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        max_val = fmaxf(max_val, val);
    }
    
    // Reduce max within warp
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    
    if (col == 0) {
        row_max[threadIdx.y] = max_val;
    }
    __syncthreads();
    
    max_val = row_max[threadIdx.y];
    
    // Compute GELU and exp(x - max)
    float sum = 0.0f;
    for (int i = col; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        float gelu_val = gelu(val);
        float exp_val = expf(gelu_val - max_val);
        output[row * cols + i] = exp_val;
        sum += exp_val;
    }
    
    // Reduce sum within warp
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (col == 0) {
        row_sum[threadIdx.y] = sum;
    }
    __syncthreads();
    
    sum = row_sum[threadIdx.y];
    
    // Normalize
    for (int i = col; i < cols; i += blockDim.x) {
        output[row * cols + i] /= sum;
    }
}

torch::Tensor fused_gelu_softmax_cuda(torch::Tensor input) {
    int rows = input.size(0);
    int cols = input.size(1);
    
    auto output = torch::empty_like(input);
    
    dim3 block(32, 8);  // 256 threads per block
    dim3 grid((rows + block.y - 1) / block.y, 1);
    
    size_t shared_mem_size = 2 * block.y * sizeof(float);
    
    fused_gelu_softmax_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
    
    return output;
}
"""

fused_gelu_softmax_cpp_source = "torch::Tensor fused_gelu_softmax_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_gelu_softmax = load_inline(
    name="fused_gelu_softmax",
    cpp_sources=fused_gelu_softmax_cpp_source,
    cuda_sources=fused_gelu_softmax_source,
    functions=["fused_gelu_softmax_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_gelu_softmax = fused_gelu_softmax

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_gelu_softmax.fused_gelu_softmax_cuda(x)
        return x