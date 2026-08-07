import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + Scale + Hardtanh + GELU
fused_gemm_scale_hardtanh_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU activation approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654;
    const float coeff = 0.044715;
    float tanh_input = sqrt_2_over_pi * (x + coeff * x * x * x);
    return x * 0.5f * (1.0f + tanhf(tanh_input));
}

__global__ void fused_gemm_scale_hardtanh_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output feature index

    if (row >= batch_size || col >= out_features) return;

    float sum = 0.0f;
    for (int k = 0; k < in_features; k++) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];
    sum *= scaling_factor;

    // Hardtanh: clamp between hardtanh_min and hardtanh_max
    sum = fmaxf(hardtanh_min, fminf(hardtanh_max, sum));

    // GELU activation
    sum = gelu(sum);

    output[row * out_features + col] = sum;
}

torch::Tensor fused_gemm_scale_hardtanh_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_dim_x = 16;
    const int block_dim_y = 16;
    dim3 block_size(block_dim_x, block_dim_y);
    dim3 grid_size((out_features + block_dim_x - 1) / block_dim_x,
                   (batch_size + block_dim_y - 1) / block_dim_y);

    fused_gemm_scale_hardtanh_gelu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor,
        hardtanh_min,
        hardtanh_max
    );

    return output;
}
"""

fused_gemm_scale_hardtanh_gelu_cpp_source = """
torch::Tensor fused_gemm_scale_hardtanh_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max);
"""

# Compile the inline CUDA kernel
fused_op = load_inline(
    name="fused_gemm_scale_hardtanh_gelu",
    cpp_sources=fused_gemm_scale_hardtanh_gelu_cpp_source,
    cuda_sources=fused_gemm_scale_hardtanh_gelu_source,
    functions=["fused_gemm_scale_hardtanh_gelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + Scale + Hardtanh + GELU in a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_gemm_scale_hardtanh_gelu_cuda(
            x,
            self.linear.weight,
            self.linear.bias,
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max
        )