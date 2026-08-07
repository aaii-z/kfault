import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-x));
        float swish = x * sigmoid_val;
        float divided = swish / 2.0f;
        float clamped1 = fmaxf(fminf(divided, 1.0f), -1.0f);
        float tanh_val = tanhf(clamped1);
        float clamped2 = fmaxf(fminf(tanh_val, 1.0f), -1.0f);
        output[idx] = clamped2;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudnn"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.fused_op = fused_op

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_op.fused_op_cuda(x)
        return x