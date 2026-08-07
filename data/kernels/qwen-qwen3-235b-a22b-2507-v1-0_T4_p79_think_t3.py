import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

__global__ void reduce_spatial_kernel(
    const float* x1,
    const float* multiplier,
    float* sum_x,
    float* sum_x2,
    int B, int C, int D, int H, int W,
    int stride_b, int stride_c, int stride_d, int stride_h, int stride_w
) {
    int c = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    float local_sum_x = 0.0f;
    float local_sum_x2 = 0.0f;

    for (int d = 0; d < D; ++d) {
        for (int h = 0; h < H; ++h) {
            for (int w = tid; w < W; w += total_threads) {
                int offset = b * stride_b + c * stride_c + d * stride_d + h * stride_h + w * stride_w;
                float val = x1[offset] * multiplier[c];
                local_sum_x += val;
                local_sum_x2 += val * val;
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
        sum_x[b * C + c] = shared_sum_x[0];
        sum_x2[b * C + c] = shared_sum_x2[0];
    }
}

__global__ void compute_mean_var_kernel(
    const float* sum_x,
    const float* sum_x2,
    float* mean,
    float* inv_std,
    int B, int C,
    int count,
    float eps
) {
    int c = blockIdx.x;
    int b = blockIdx.y;
    int idx = b * C + c;

    float sum_val = sum_x[idx];
    float sum_val2 = sum_x2[idx];

    float mean_val = sum_val / count;
    float var_val = sum_val2 / count - mean_val * mean_val;
    if (var_val < 0.0f) var_val = 0.0f;
    float inv_std_val = 1.0f / sqrtf(var_val + eps);

    mean[idx] = mean_val;
    inv_std[idx] = inv_std_val;
}

__global__ void main_fused_kernel(
    const float* x1,
    const float* multiplier,
    const float* mean,
    const float* inv_std,
    float* out,
    float clamp_min,
    float clamp_max,
    int B, int C, int D, int H, int W,
    int stride_b, int stride_c, int stride_d, int stride_h, int stride_w
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
        int offset = b * stride_b + c * stride_c + d * stride_d + h * stride_h + w * stride_w;
        float x2 = x1[offset] * multiplier[c];
        int idx_mean = b * C + c;
        float normalized = (x2 - mean[idx_mean]) * inv_std[idx_mean];
        float x3 = normalized;
        float x4 = fmaxf(fminf(x3, clamp_max), clamp_min);
        float x5 = x4 * multiplier[c];

        if (x5 > max_val) {
            max_val = x5;
        }
    }

    out[idx] = max_val;
}

torch::Tensor fused_op_cuda(
    torch::Tensor x1,
    torch::Tensor multiplier,
    float eps,
    float clamp_min,
    float clamp_max,
    int stride_b,
    int stride_c,
    int stride_d,
    int stride_h,
    int stride_w
) {
    auto B = x1.size(0);
    auto C = x1.size(1);
    auto D = x1.size(2);
    auto H = x1.size(3);
    auto W = x1.size(4);
    int count = D * H * W;

    auto sum_x = torch::zeros({B, C}, x1.options());
    auto sum_x2 = torch::zeros({B, C}, x1.options());
    auto mean = torch::zeros({B, C}, x1.options());
    auto inv_std = torch::zeros({B, C}, x1.options());

    dim3 block_reduce(256);
    dim3 grid_reduce(C, B);
    reduce_spatial_kernel<<<grid_reduce, block_reduce>>>(
        x1.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        sum_x.data_ptr<float>(),
        sum_x2.data_ptr<float>(),
        B, C, D, H, W,
        stride_b, stride_c, stride_d, stride_h, stride_w
    );

    compute_mean_var_kernel<<<grid_reduce, dim3(1)>>>(
        sum_x.data_ptr<float>(),
        sum_x2.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        B, C,
        count,
        eps
    );

    auto out = torch::zeros({B, D, H, W}, x1.options());
    int total_elements = B * D * H * W;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    main_fused_kernel<<<grid_size, block_size>>>(
        x1.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        out.data_ptr<float>(),
        clamp_min,
        clamp_max,
        B, C, D, H, W,
        stride_b, stride_c, stride_d, stride_h, stride_w
    );

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_op_cuda("
    "torch::Tensor x1, torch::Tensor multiplier, "
    "float eps, float clamp_min, float clamp_max, "
    "int stride_b, int stride_c, int stride_d, int stride_h, int stride_w);"
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
            self.multiplier.squeeze().contiguous(), 
            self.instance_norm.eps,
            self.clamp_min,
            self.clamp_max,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            x.stride(4)
        )
        return x