import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_along_dim_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

__global__ void min_along_dim_kernel(
    const float* input, 
    float* output, 
    int batch, 
    int channels, 
    int D, 
    int H, 
    int W,
    int s0, int s1, int s2, int s3, int s4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch * channels * H * W;
    if (idx >= total_threads) return;

    int b = idx / (channels * H * W);
    int rem = idx % (channels * H * W);
    int c = rem / (H * W);
    rem = rem % (H * W);
    int h = rem / W;
    int w = rem % W;

    int base = b * s0 + c * s1 + h * s3 + w * s4;
    float min_val = FLT_MAX;

    for (int d = 0; d < D; d++) {
        float val = input[base + d * s2];
        if (val < min_val) {
            min_val = val;
        }
    }

    int out_idx = b * (channels * H * W) + c * (H * W) + h * W + w;
    output[out_idx] = min_val;
}

torch::Tensor min_along_dim_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 5, "Input must be 5D tensor");
    int batch = input.size(0);
    int channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto strides = input.strides();
    int s0 = strides[0];
    int s1 = strides[1];
    int s2 = strides[2];
    int s3 = strides[3];
    int s4 = strides[4];

    auto output = torch::empty({batch, channels, H, W}, input.options());

    int total_threads = batch * channels * H * W;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    min_along_dim_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, D, H, W,
        s0, s1, s2, s3, s4
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

min_along_dim_cpp = "torch::Tensor min_along_dim_cuda(torch::Tensor input);"

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <vector>

__global__ void softmax_kernel(
    const float* input, 
    float* output, 
    int batch, 
    int channels, 
    int H, 
    int W,
    int s0, int s1, int s2, int s3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch * H * W;
    if (idx >= total_threads) return;

    int b = idx / (H * W);
    int hw = idx % (H * W);
    int h = hw / W;
    int w = hw % W;

    int base = b * s0 + h * s2 + w * s3;
    float max_val = -FLT_MAX;

    for (int c = 0; c < channels; c++) {
        float val = input[base + c * s1];
        if (val > max_val) {
            max_val = val;
        }
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < channels; c++) {
        float val = input[base + c * s1] - max_val;
        float exp_val = expf(val);
        sum_exp += exp_val;
    }

    if (sum_exp == 0.0f) {
        float uniform = 1.0f / channels;
        for (int c = 0; c < channels; c++) {
            output[base + c * s1] = uniform;
        }
        return;
    }

    for (int c = 0; c < channels; c++) {
        float val = input[base + c * s1] - max_val;
        float exp_val = expf(val);
        output[base + c * s1] = exp_val / sum_exp;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor");
    int batch = input.size(0);
    int channels = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto strides = input.strides();
    int s0 = strides[0];
    int s1 = strides[1];
    int s2 = strides[2];
    int s3 = strides[3];

    auto output = torch::empty_like(input);

    int total_threads = batch * H * W;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, H, W,
        s0, s1, s2, s3
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

softmax_cpp = "torch::Tensor softmax_cuda(torch::Tensor input);"

min_along_dim = load_inline(
    name="min_along_dim",
    cpp_sources=min_along_dim_cpp,
    cuda_sources=min_along_dim_source,
    functions=["min_along_dim_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernels for min along depth dimension and softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.min_along_dim = min_along_dim
        self.softmax = softmax

    def forward(self, x):
        x = self.conv(x)
        x = self.min_along_dim.min_along_dim_cuda(x)
        x = self.softmax.softmax_cuda(x)
        return x