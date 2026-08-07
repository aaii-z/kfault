import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_layernorm_gelu_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu_approx(float x) {
    float c = 0.044715f;
    float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + c * x * x * x)));
}

__global__ void fused_layernorm_gelu_scale_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    float scaling_factor,
    int C,
    int total_spatial,
    int spatial_size,
    float eps
) {
    int idx = blockIdx.x;
    if (idx >= total_spatial) return;

    int tid = threadIdx.x;
    if (tid >= C) return;

    float x = input[tid * spatial_size + idx];
    
    extern __shared__ float s_data[];
    float* sum = s_data;
    float* sum_sq = s_data + C;

    sum[tid] = x;
    sum_sq[tid] = x * x;
    __syncthreads();

    for (int stride = C / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum[tid] += sum[tid + stride];
            sum_sq[tid] += sum_sq[tid + stride];
        }
        __syncthreads();
    }

    float local_mean = 0.0f;
    float local_variance = 0.0f;
    if (tid == 0) {
        local_mean = sum[0] / C;
        local_variance = sum_sq[0] / C - local_mean * local_mean;
    }
    __syncthreads();

    if (tid == 0) {
        sum[0] = local_mean;
        sum[1] = local_variance;
    }
    __syncthreads();
    local_mean = sum[0];
    local_variance = sum[1];
    __syncthreads();

    float inv_std = 1.0f / sqrtf(local_variance + eps);
    float norm_val = (x - local_mean) * inv_std * gamma[tid] + beta[tid];
    output[tid * spatial_size + idx] = gelu_approx(norm_val) * scaling_factor;
}

torch::Tensor fused_layernorm_gelu_scale_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float scaling_factor,
    float eps
) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int total_spatial = B * D * H * W;
    int spatial_size = D * H * W;

    auto output = torch::empty_like(input);
    const int block_size = C;
    const int num_blocks = total_spatial;
    size_t shared_mem = 2 * C * sizeof(float);

    fused_layernorm_gelu_scale_kernel<<<num_blocks, block_size, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        scaling_factor,
        C,
        total_spatial,
        spatial_size,
        eps
    );

    return output;
}
"""

fused_layernorm_gelu_scale_cpp = (
    "torch::Tensor fused_layernorm_gelu_scale_cuda("
    "torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, "
    "float scaling_factor, float eps);"
)

fused_op = load_inline(
    name="fused_layernorm_gelu_scale",
    cpp_sources=fused_layernorm_gelu_scale_cpp,
    cuda_sources=fused_layernorm_gelu_scale_source,
    functions=["fused_layernorm_gelu_scale_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        self.eps = eps
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_layernorm_gelu_scale_cuda(
            x, 
            self.layer_norm.weight, 
            self.layer_norm.bias, 
            self.scaling_factor, 
            self.eps
        )
        return x