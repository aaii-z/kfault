import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused GEMM + BiasAdd + Hardtanh + Mish
fused_gemm_act_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Hardtanh clamp values
const float hardtanh_min_val = -1.0f;
const float hardtanh_max_val = 1.0f;

// Mish activation function
__device__ float mish(float x) {
    float sp = log1pf(expf(x));
    return x * tanhf(sp);
}

// Kernel for BiasAdd + Hardtanh + Mish
__global__ void bias_act_kernel(
    float* output,
    const float* bias,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int bias_idx = idx % out_features;
        float val = output[idx] + bias[bias_idx];
        val = fmaxf(hardtanh_min_val, fminf(hardtanh_max_val, val));
        output[idx] = mish(val);
    }
}

torch::Tensor fused_gemm_bias_hardtanh_mish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::zeros({batch_size, out_features}, input.options());

    // Use cuBLAS for efficient GEMM
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_features, batch_size, in_features,
        &alpha,
        weight.data_ptr<float>(), in_features,
        input.data_ptr<float>(), in_features,
        &beta,
        output.data_ptr<float>(), out_features
    );

    cublasDestroy(handle);

    // Launch activation kernel
    const int threads = 256;
    const int blocks = (batch_size * out_features + threads - 1) / threads;
    bias_act_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

fused_gemm_act_cpp_source = """
torch::Tensor fused_gemm_bias_hardtanh_mish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

# Load the custom CUDA extension
fused_gemm_act = load_inline(
    name="fused_gemm_act",
    cpp_sources=fused_gemm_act_cpp_source,
    cuda_sources=fused_gemm_act_source,
    functions=["fused_gemm_bias_hardtanh_mish"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.fused_gemm_act = fused_gemm_act

    def forward(self, x):
        x = self.fused_gemm_act.fused_gemm_bias_hardtanh_mish(x, self.weight, self.bias)
        x = self.groupnorm(x)
        return x