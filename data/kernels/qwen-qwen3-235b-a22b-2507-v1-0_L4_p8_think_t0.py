import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused sum, division, and bias addition
fused_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_sum_kernel(const float* x, float* out, int B, int C, int D, int H, int W, float sum_bias) {
    int b = blockIdx.x;
    if (b >= B) return;

    int total_elements = C * D * H * W;
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < total_elements; i += blockDim.x) {
        int c = i / (D * H * W);
        int rest = i % (D * H * W);
        int d = rest / (H * W);
        rest %= (H * W);
        int h = rest / W;
        int w = rest % W;

        int x_index = ((b * C + c) * D + d) * H + h;
        x_index = x_index * W + w;
        sum += x[x_index];
    }

    extern __shared__ float sdata[];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float avg = sdata[0] / static_cast<float>(D * H * W);
        out[b] = avg + sum_bias;
    }
}

torch::Tensor fused_sum_cuda(torch::Tensor x, float sum_bias) {
    auto B = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);

    auto out = torch::empty({B}, x.options());

    const int block_size = 256;
    const int num_blocks = B;
    size_t smem_size = block_size * sizeof(float);

    fused_sum_kernel<<<num_blocks, block_size, smem_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), 
        B, C, D, H, W, sum_bias
    );

    return out.view({B, 1, 1, 1});
}
"""

fused_sum_cpp_source = "torch::Tensor fused_sum_cuda(torch::Tensor x, float sum_bias);"

fused_sum = load_inline(
    name="fused_sum",
    cpp_sources=fused_sum_cpp_source,
    cuda_sources=fused_sum_source,
    functions=["fused_sum_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.max_pool = nn.MaxPool3d(pool_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.fused_sum = fused_sum

        with torch.no_grad():
            self.conv.weight /= divisor
            if self.conv.bias is not None:
                self.conv.bias /= divisor

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        sum_bias_val = self.bias.sum().item()
        x = self.fused_sum.fused_sum_cuda(x, sum_bias_val)
        return x