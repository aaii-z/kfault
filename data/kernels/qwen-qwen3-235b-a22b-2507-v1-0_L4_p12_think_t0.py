import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_gemm_mult_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_mult_leakyrelu_kernel(const float* input, float* output, int size, float multiplier, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        float result = val * multiplier;
        output[idx] = result >= 0 ? result : result * negative_slope;
    }
}

torch::Tensor fused_gemm_mult_leakyrelu_cuda(torch::Tensor input, float multiplier, float negative_slope) {
    auto output = torch::empty_like(input);
    int size = input.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_gemm_mult_leakyrelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        multiplier,
        negative_slope
    );

    return output;
}
"""

fused_gemm_mult_leakyrelu_cpp = "torch::Tensor fused_gemm_mult_leakyrelu_cuda(torch::Tensor input, float multiplier, float negative_slope);"

fused_op = load_inline(
    name="fused_gemm_mult_leakyrelu",
    cpp_sources=fused_gemm_mult_leakyrelu_cpp,
    cuda_sources=fused_gemm_mult_leakyrelu_source,
    functions=["fused_gemm_mult_leakyrelu_cuda"],
    verbose=False,
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses multiplication and LeakyReLU into a single CUDA kernel after Gemm.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.fused_op = fused_op

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_gemm_mult_leakyrelu_cuda(x, self.multiplier, self.negative_slope)
        return x