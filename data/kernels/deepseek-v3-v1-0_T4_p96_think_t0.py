import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void max_pool3d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    int pool_size,
    int out_depth,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_depth * out_height * out_width) return;

    int n = idx / (channels * out_depth * out_height * out_width);
    int c = (idx / (out_depth * out_height * out_width)) % channels;
    int d = (idx / (out_height * out_width)) % out_depth;
    int h = (idx / out_width) % out_height;
    int w = idx % out_width;

    int start_d = d * pool_size;
    int start_h = h * pool_size;
    int start_w = w * pool_size;

    float max_val = -FLT_MAX;
    for (int pd = 0; pd < pool_size; ++pd) {
        for (int ph = 0; ph < pool_size; ++ph) {
            for (int pw = 0; pw < pool_size; ++pw) {
                int in_d = start_d + pd;
                int in_h = start_h + ph;
                int in_w = start_w + pw;
                if (in_d < depth && in_h < height && in_w < width) {
                    int input_idx = ((n * channels + c) * depth + in_d) * height * width + in_h * width + in_w;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
    }
    output[idx] = max_val;
}

__global__ void global_avg_pool_clamp_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    float clamp_min,
    float clamp_max
) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int total_elements = depth * height * width;

    extern __shared__ float shared[];
    float* shmem = shared;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        int d = i / (height * width);
        int h = (i / width) % height;
        int w = i % width;
        int idx = ((n * channels + c) * depth + d) * height * width + h * width + w;
        sum += input[idx];
    }

    shmem[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shmem[threadIdx.x] += shmem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float avg = shmem[0] / total_elements;
        output[n * channels + c] = fminf(fmaxf(avg, clamp_min), clamp_max);
    }
}

torch::Tensor fused_max_pool3d(
    torch::Tensor input,
    int pool_size
) {
    CHECK_INPUT(input);
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    int out_depth = depth / pool_size;
    int out_height = height / pool_size;
    int out_width = width / pool_size;

    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, input.options());

    int total_elements = batch_size * channels * out_depth * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    max_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        pool_size,
        out_depth,
        out_height,
        out_width
    );

    return output;
}

torch::Tensor fused_global_avg_pool_clamp(
    torch::Tensor input,
    float clamp_min,
    float clamp_max
) {
    CHECK_INPUT(input);
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    auto output = torch::empty({batch_size, channels, 1, 1, 1}, input.options());

    dim3 blocks(batch_size, channels);
    int threads = 1024;
    int shmem_size = threads * sizeof(float);

    global_avg_pool_clamp_kernel<<<blocks, threads, shmem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        clamp_min,
        clamp_max
    );

    return output;
}
"""

cpp_source = """
torch::Tensor fused_max_pool3d(torch::Tensor input, int pool_size);
torch::Tensor fused_global_avg_pool_clamp(torch::Tensor input, float clamp_min, float clamp_max);
"""

custom_ops = load_inline(
    name='custom_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_max_pool3d', 'fused_global_avg_pool_clamp'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale
        x = custom_ops.fused_max_pool3d(x, self.maxpool_kernel_size)
        x = custom_ops.fused_global_avg_pool_clamp(x, self.clamp_min, self.clamp_max)
        return x