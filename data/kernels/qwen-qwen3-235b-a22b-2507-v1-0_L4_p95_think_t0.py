import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_activation_kernel(
    const float* matmul_result,
    const float* add_value,
    float* output,
    int out_features,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int col = idx % out_features;
    float x = matmul_result[idx] + add_value[col];

    // Swish: x * sigmoid(x)
    x = x * (1.0f / (1.0f + expf(-x)));

    // Tanh
    x = tanhf(x);

    // GELU approximation
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    float cdf = 0.5f * (1.0f + tanhf(inner));
    x = x * cdf;

    // Hardtanh: clamp to [-1, 1]
    x = fmaxf(-1.0f, fminf(1.0f, x));

    output[idx] = x;
}

torch::Tensor fused_activation_cuda(
    torch::Tensor matmul_result,
    torch::Tensor add_value
) {
    auto batch_size = matmul_result.size(0);
    auto out_features = matmul_result.size(1);
    auto total_elements = batch_size * out_features;
    auto output = torch::empty_like(matmul_result);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(
        matmul_result.data_ptr<float>(),
        add_value.data_ptr<float>(),
        output.data_ptr<float>(),
        out_features,
        total_elements
    );

    return output;
}
"""

fused_activation_cpp = "torch::Tensor fused_activation_cuda(torch::Tensor matmul_result, torch::Tensor add_value);"

fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses addition and multiple activation functions into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_activation.fused_activation_cuda(x, self.add_value)
        return x