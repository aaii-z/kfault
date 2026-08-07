import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

__global__ void reduce_channel_sum_kernel(
    const float* x0,
    const float* multiplier,
    float* sum_x,
    float* sum_x2,
    int B, int C, int D, int H, int W
) {
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    float local_sum_x = 0.0f;
    float local_sum_x2 = 0.0f;

    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            for (int h = 0; h < H; ++h) {
                for (int w = tid; w < W; w += total_threads) {
                    int offset = ((b * C + c) * D + d) * H + h) * W + w;
                    float val = x0[offset] * multiplier[c];
                    local_sum_x += val;
                    local_sum_x2 += val * val;
                }
            }
        }
    }

    __shared__ float shared_sum_x[256];
    __shared__ float shared_sum_x2[256];
    shared_sum_x[tid] = local_sum_x;
    shared_sum_x2[tid] = local_sum_x2;
    __syncthreads();

    for (int s = total_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum_x[tid] += shared_sum_x[tid + s];
            shared_sum_x2[tid] += shared_sum_x2[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum_x[c] = shared_sum_x[0];
        sum_x2[c] = shared_sum_x2[0];
    }
}

__global__ void compute_mean_var_kernel(
    const float* sum_x,
    const float* sum_x2,
    float* mean,
    float* var,
    int C,
    int count,
    float eps
) {
    int c = blockIdx.x;
    if (c >= C) return;

    mean[c] = sum_x[c] / count;
    float variance = sum_x2[c] / count - mean[c] * mean[c];
    var[c] = fmaxf(variance, 0.0f);
}

__global__ void fused_op_main_kernel(
    const float* x0,
    const float* multiplier,
    const float* gamma,
    const float* beta,
    const float* mean,
    const float* var,
    float* out,
    float eps_val,
    float clamp_min,
    float clamp_max,
    int B, int C, int D, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * D * H * W;
    if (idx >= total_elements) return;

    int b = idx / (D * H * W);
    int dhw = idx % (D * H * W);
    int d = dhw / (H * W);
    int hw = dhw % (H * W);
    int h = hw / W;
    int w = hw % W;

    float max_val = -FLT_MAX;

    for (int c = 0; c < C; ++c) {
        int offset = ((b * C + c) * D + d) * H + h) * W + w;
        float x1 = x0[offset] * multiplier[c];
        float normalized = (x1 - mean[c]) * rsqrtf(var[c] + eps_val);
        float x2 = normalized * gamma[c] + beta[c];
        float x3 = fmaxf(fminf(x2, clamp_max), clamp_min);
        float x4 = x3 * multiplier[c];

        if (x4 > max_val) {
            max_val = x4;
        }
    }

    out[idx] = max_val;
}

torch::Tensor fused_op_cuda(
    torch::Tensor x0,
    torch::Tensor multiplier,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    float clamp_min,
    float clamp_max
) {
    auto B = x0.size(0);
    auto C = x0.size(1);
    auto D = x0.size(2);
    auto H = x0.size(3);
    auto W = x0.size(4);
    int count = D * H * W;

    auto sum_x = torch::zeros({C}, x0.options());
    auto sum_x2 = torch::zeros({C}, x0.options());
    auto mean = torch::zeros({C}, x0.options());
    auto var = torch::zeros({C}, x0.options());

    int block_size = 256;
    reduce_channel_sum_kernel<<<C, block_size>>>(
        x0.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        sum_x.data_ptr<float>(),
        sum_x2.data_ptr<float>(),
        B, C, D, H, W
    );

    compute_mean_var_kernel<<<C, 1>>>(
        sum_x.data_ptr<float>(),
        sum_x2.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        C,
        count,
        eps
    );

    auto out = torch::zeros({B, D, H, W}, x0.options());
    int total_elements = B * D * H * W;
    int grid_size = (total_elements + block_size - 1) / block_size;
    fused_op_main_kernel<<<grid_size, block_size>>>(
        x0.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        out.data_ptr<float>(),
        eps,
        clamp_min,
        clamp_max,
        B, C, D, H, W
    );

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda("
    "torch::Tensor x0, torch::Tensor multiplier, torch::Tensor gamma, torch::Tensor beta, "
    "float eps, float clamp_min, float clamp_max);"
)

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_op_cuda(
            x, 
            self.multiplier.squeeze(), 
            self.instance_norm.weight, 
            self.instance_norm.bias,
            self.instance_norm.eps,
            self.clamp_min,
            self.clamp_max
        )
        return x