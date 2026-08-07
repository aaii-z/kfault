import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fused_sigmoid_scaling_residual_kernel(
    const float* input, float* output, float scaling_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = scaling_factor * sigmoid_x + x;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_sigmoid_scaling_residual_kernel<<<num_blocks, block_size, 0, stream>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        scaling_factor, 
        size
    );

    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, float scaling_factor);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model implementing fused CUDA kernel for Sigmoid_Scaling_ResidualAdd pattern.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_op

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_op_cuda(x, self.scaling_factor)
        return x