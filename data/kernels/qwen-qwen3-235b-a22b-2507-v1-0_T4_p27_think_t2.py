import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardswish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    }
}

torch::Tensor hardswish_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    hardswish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

hardswish_cpp = "torch::Tensor hardswish_cuda(torch::Tensor input);"

mean_pool_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_pool_3d_kernel(
    const float* input,
    float* output,
    int B, int C, int D, int H, int W
) {
    int b = blockIdx.y;
    int c = blockIdx.x;
    if (b >= B || c >= C) return;

    int total = D * H * W;
    float sum = 0.0f;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < total; i += stride) {
        int d = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int idx = ((b * C + c) * D + d) * H * W + h * W + w;
        sum += input[idx];
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b * C + c] = shared_sum[0] / total;
    }
}

torch::Tensor mean_pool_3d_cuda(torch::Tensor input) {
    auto B = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto output = torch::empty({B, C}, input.options());

    dim3 block_dim(256);
    dim3 grid_dim(C, B);

    mean_pool_3d_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W
    );

    return output;
}
"""

mean_pool_3d_cpp = "torch::Tensor mean_pool_3d_cuda(torch::Tensor input);"

hardswish_op = load_inline(
    name="hardswish_op",
    cpp_sources=hardswish_cpp,
    cuda_sources=hardswish_source,
    functions=["hardswish_cuda"],
    verbose=False
)

mean_pool_3d_op = load_inline(
    name="mean_pool_3d_op",
    cpp_sources=mean_pool_3d_cpp,
    cuda_sources=mean_pool_3d_source,
    functions=["mean_pool_3d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.hardswish = hardswish_op.hardswish_cuda
        self.mean_pool_3d = mean_pool_3d_op.mean_pool_3d_cuda

    def forward(self, x):
        x = self.conv(x)
        x = self.hardswish(x)
        x = self.group_norm(x)
        x = self.mean_pool_3d(x)
        return x