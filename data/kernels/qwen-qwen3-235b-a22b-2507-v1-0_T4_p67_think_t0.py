import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GELU and global average pooling
gelu_gap_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void gelu_gap_kernel(const float* input, float* output, int B, int C, int H, int W) {
    int idx = blockIdx.x;
    int b = idx / C;
    int c = idx % C;
    if (b >= B || c >= C) return;

    int total_elements = H * W;
    float sum = 0.0f;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int i = tid; i < total_elements; i += num_threads) {
        int h = i / W;
        int w = i % W;
        int input_idx = ((b * C + c) * H + h) * W + w;
        float x = input[input_idx];

        // Compute GELU using standard approximation
        float x3 = x * x * x;
        float inner = x + 0.044715f * x3;
        float tanh_val = tanhf(0.7978845608028654f * inner);
        float gelu_val = 0.5f * x * (1.0f + tanh_val);

        sum += gelu_val;
    }

    __shared__ float shared_sum[BLOCK_SIZE];
    shared_sum[tid] = sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b * C + c] = shared_sum[0] / total_elements;
    }
}

torch::Tensor gelu_gap_cuda(torch::Tensor input) {
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::zeros({B, C}, input.options());

    int grid_size = B * C;
    dim3 grid(grid_size);
    dim3 block(BLOCK_SIZE);

    gelu_gap_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), B, C, H, W);

    return output;
}
"""

gelu_gap_cpp_source = "torch::Tensor gelu_gap_cuda(torch::Tensor input);"

# Compile the inline CUDA code
gelu_gap = load_inline(
    name="gelu_gap",
    cpp_sources=gelu_gap_cpp_source,
    cuda_sources=gelu_gap_source,
    functions=["gelu_gap_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.gelu_gap = gelu_gap

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu_gap.gelu_gap_cuda(x)
        return x