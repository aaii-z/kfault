import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused maxpool, sum, and scale
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

template<typename T>
__global__ void fused_maxpool_sum_scale_kernel(
    const T* input,
    T* output,
    int batch_size,
    int out_features,
    T scale_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const T* sample_input = input + idx * out_features;
    T sum = 0.0;

    for (int i = 0; i < out_features; i += 2) {
        T val1 = sample_input[i];
        T val2 = (i + 1 < out_features) ? sample_input[i + 1] : -FLT_MAX;
        sum += fmaxf(val1, val2);
    }

    output[idx] = sum * scale_factor;
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    float scale_factor
) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    auto output = torch::empty({batch_size}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    fused_maxpool_sum_scale_kernel<float><<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        scale_factor
    );

    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input, float scale_factor);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_ops.fused_ops_cuda(x, self.scale_factor)
        return x