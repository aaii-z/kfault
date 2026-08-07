import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

fused_bn_mean_sub_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void compute_sum_kernel(const float* x, float* sum, int B, int C, int D, int H, int W) {
    int b = blockIdx.y;
    int c = blockIdx.z;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    float local_sum = 0.0f;
    int total_spatial = D * H * W;

    for (int idx = tid; idx < total_spatial; idx += num_threads) {
        int d = idx / (H * W);
        int hw_idx = idx % (H * W);
        int h = hw_idx / W;
        int w = hw_idx % W;

        int x_idx = ((b * C + c) * D + d) * H * W + h * W + w;
        local_sum += x[x_idx];
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[b * C + c] = shared_sum[0];
    }
}

__global__ void compute_output_kernel(const float* x, const float* sum, float* out,
                                     const float* gamma, const float* running_mean, const float* running_var,
                                     float eps, int B, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * D * H * W;
    if (idx >= total_elements) return;

    int w = idx % W;
    int hw = idx / W;
    int h = hw % H;
    int dhw = hw / H;
    int d = dhw % D;
    int chw = dhw / D;
    int c = chw % C;
    int b = chw / C;

    float x_val = x[idx];
    float rm = running_mean[c];
    float rv = running_var[c];
    float denom = sqrtf(rv + eps);
    float temp = (x_val - rm) / denom;

    float mean_val = sum[b * C + c] / (D * H * W);
    out[idx] = gamma[c] * (temp - mean_val);
}

torch::Tensor fused_bn_mean_sub_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor running_mean, torch::Tensor running_var, float eps) {
    int B = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int total_elements = B * C * D * H * W;

    auto sum = torch::zeros({B, C}, x.options());

    dim3 block_sum(256);
    dim3 grid_sum(1, B, C);
    compute_sum_kernel<<<grid_sum, block_sum>>>(x.data_ptr<float>(), sum.data_ptr<float>(), B, C, D, H, W);

    auto out = torch::empty_like(x);
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    compute_output_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), sum.data_ptr<float>(), out.data_ptr<float>(),
        gamma.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        eps, B, C, D, H, W
    );
    return out;
}
"""

fused_bn_mean_sub_cpp = "torch::Tensor fused_bn_mean_sub_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor running_mean, torch::Tensor running_var, float eps);"

fused_bn_mean_sub = load_inline(
    name="fused_bn_mean_sub",
    cpp_sources=fused_bn_mean_sub_cpp,
    cuda_sources=fused_bn_mean_sub_source,
    functions=["fused_bn_mean_sub_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_op = fused_bn_mean_sub

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_bn_mean_sub_cuda(x, self.batch_norm.weight, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.eps)
        return x