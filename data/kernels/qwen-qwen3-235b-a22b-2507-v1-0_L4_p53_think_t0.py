import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel(const float* input, float* output, int size, float scaling_factor, float hardtanh_min, float hardtanh_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] * scaling_factor;
        x = (x < hardtanh_min) ? hardtanh_min : (x > hardtanh_max) ? hardtanh_max : x;
        float x3 = x * x * x;
        float inner = 0.044715f * x3 + x;
        float tanh_val = tanhf(0.7978845608028654f * inner);
        float gelu = 0.5f * x * (1.0f + tanh_val);
        output[idx] = gelu;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, float scaling_factor, float hardtanh_min, float hardtanh_max) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_op_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        scaling_factor,
        hardtanh_min,
        hardtanh_max
    );

    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, float scaling_factor, float hardtanh_min, float hardtanh_max);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_op = fused_op

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_op_cuda(x, self.scaling_factor, self.hardtanh_min, self.hardtanh_max)
        return x