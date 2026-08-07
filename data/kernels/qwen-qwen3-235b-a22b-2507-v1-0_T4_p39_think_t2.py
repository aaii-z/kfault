import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_gemm_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x)

__global__ void fused_gemm_scale_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    const float* scale,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int i = idx / out_features;
    int j = idx % out_features;

    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += x[i * in_features + k] * weight[j * in_features + k];
    }
    output[idx] = (sum + bias[j]) * scale[j];
}

torch::Tensor fused_gemm_scale_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(scale);

    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, x.options());

    int total_elements = batch_size * out_features;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_gemm_scale_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_gemm_scale_cpp_source = (
    "torch::Tensor fused_gemm_scale_cuda("
    "torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor scale);"
)

fused_gemm_scale = load_inline(
    name="fused_gemm_scale",
    cpp_sources=fused_gemm_scale_cpp_source,
    cuda_sources=fused_gemm_scale_source,
    functions=["fused_gemm_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.fused_gemm_scale = fused_gemm_scale

    def forward(self, x):
        x = self.fused_gemm_scale.fused_gemm_scale_cuda(x, self.gemm.weight, self.gemm.bias, self.scale)
        x = self.bn(x)
        return x