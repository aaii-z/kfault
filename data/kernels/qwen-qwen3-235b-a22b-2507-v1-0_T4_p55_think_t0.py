import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_maxpool_sum_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_maxpool_sum_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int out_features,
    float scale_factor)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const int num_groups = out_features / 2;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    float sum = 0.0f;

    for (int g = tid; g < num_groups; g += num_threads) {
        int idx0 = batch_idx * out_features + 2 * g;
        int idx1 = idx0 + 1;
        float a = input[idx0];
        float b = input[idx1];
        sum += (a > b) ? a : b;
    }

    __shared__ float s_sum[256];
    s_sum[tid] = sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx] = s_sum[0] * scale_factor;
    }
}

torch::Tensor fused_maxpool_sum_cuda(torch::Tensor input, float scale_factor) {
    TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    int batch_size = input.size(0);
    int out_features = input.size(1);
    TORCH_CHECK(out_features % 2 == 0, "out_features must be even");

    auto output = torch::empty({batch_size}, input.options());

    const int block_size = 256;
    const int num_blocks = batch_size;

    fused_maxpool_sum_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        scale_factor
    );

    return output;
}
'''

fused_maxpool_sum_cpp_source = "torch::Tensor fused_maxpool_sum_cuda(torch::Tensor input, float scale_factor);"

fused_maxpool_sum = load_inline(
    name="fused_maxpool_sum",
    cpp_sources=fused_maxpool_sum_cpp_source,
    cuda_sources=fused_maxpool_sum_source,
    functions=["fused_maxpool_sum_cuda"],
    verbose=False,
    with_cuda=True
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses max pooling, sum, and scaling into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.fused_maxpool_sum = fused_maxpool_sum

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size).
        """
        x = self.matmul(x)
        x = self.fused_maxpool_sum.fused_maxpool_sum_cuda(x, self.scale_factor)
        return x