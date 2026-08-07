import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

global_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_mean_kernel(const float* x, float* out, int batch_size, int C, int D, int H, int W) {
    int total_elements_per_batch = C * D * H * W;
    int idx = blockIdx.x;

    if (idx >= batch_size) return;

    float sum = 0.0f;
    int start_idx = idx * total_elements_per_batch;

    for (int i = threadIdx.x; i < total_elements_per_batch; i += blockDim.x) {
        sum += x[start_idx + i];
    }

    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[idx] = sdata[0] / total_elements_per_batch;
    }
}

torch::Tensor global_mean_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    int total_elements_per_batch = C * D * H * W;

    auto out = torch::empty({batch_size, 1}, x.options());

    const int block_size = 256;
    const int num_blocks = batch_size;

    global_mean_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        C,
        D,
        H,
        W
    );

    return out;
}
"""

global_mean_cpp_source = "torch::Tensor global_mean_cuda(torch::Tensor x);"

global_mean = load_inline(
    name="global_mean",
    cpp_sources=global_mean_cpp_source,
    cuda_sources=global_mean_source,
    functions=["global_mean_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.global_mean = global_mean

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = x.contiguous()
        x = self.global_mean.global_mean_cuda(x)
        return x