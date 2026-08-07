import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define THREADS_PER_BLOCK 256

__global__ void mean_reduction_kernel(
    const float* input,
    float* output,
    int B,
    int C,
    int D,
    int H,
    int W
) {
    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int spatial_size = D * H * W;
    
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int d = i / (H * W);
        int h = (i % (H * W)) / W;
        int w = i % W;
        int idx = ((batch * C + channel) * D + d) * H * W + h * W + w;
        sum += input[idx];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch * C + channel] = sdata[0] / spatial_size;
    }
}

torch::Tensor mean_reduction_cuda(torch::Tensor input) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    auto output = torch::zeros({B, C}, input.options());
    
    dim3 blocks(B, C);
    int threads = THREADS_PER_BLOCK;
    int shared_mem = threads * sizeof(float);
    
    mean_reduction_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor mean_reduction_cuda(torch::Tensor input);
"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["mean_reduction_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.hardswish(x)
        x = self.group_norm(x)
        x = custom_ops.mean_reduction_cuda(x)
        return x