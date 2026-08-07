import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_linear_div_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu(float x) {
    const float kAlpha = 0.7978845608028654f;
    const float kBeta = 0.044715f;
    float x3 = x * x * x;
    float inner = kAlpha * (x + kBeta * x3);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

__global__ void fused_linear_div_gelu_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    float divisor,
    int batch_size,
    int input_size,
    int output_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * output_size) return;

    int i = idx / output_size;
    int j = idx % output_size;

    float sum = 0.0f;
    int k = 0;
    for (; k <= input_size - 4; k += 4) {
        float4 x4 = *(float4*)(x + i * input_size + k);
        float4 w4 = *(float4*)(weight + j * input_size + k);
        sum += x4.x * w4.x + x4.y * w4.y + x4.z * w4.z + x4.w * w4.w;
    }
    for (; k < input_size; ++k) {
        sum += x[i * input_size + k] * weight[j * input_size + k];
    }
    sum += bias[j];
    sum /= divisor;
    out[idx] = gelu(sum);
}

torch::Tensor fused_linear_div_gelu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.device().is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(weight.size(0) == bias.size(0), "weight and bias size mismatch");

    int batch_size = x.size(0);
    int input_size = x.size(1);
    int output_size = weight.size(0);

    auto out = torch::empty({batch_size, output_size}, x.options());

    int total_elements = batch_size * output_size;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_linear_div_gelu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        divisor,
        batch_size,
        input_size,
        output_size
    );

    return out;
}
"""

fused_linear_div_gelu_cpp = (
    "torch::Tensor fused_linear_div_gelu_cuda("
    "torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float divisor);"
)

fused_linear_div_gelu = load_inline(
    name="fused_linear_div_gelu",
    cpp_sources=fused_linear_div_gelu_cpp,
    cuda_sources=fused_linear_div_gelu_source,
    functions=["fused_linear_div_gelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_op = fused_linear_div_gelu

    def forward(self, x):
        return self.fused_op.fused_linear_div_gelu_cuda(
            x, self.linear.weight, self.linear.bias, self.divisor
        )