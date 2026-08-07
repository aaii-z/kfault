import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hardtanh_mean_tanh_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__device__ float hardtanh(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__global__ void hardtanh_mean_tanh_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float hardtanh_min,
    float hardtanh_max
) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    
    int spatial_size = height * width;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int h = i / width;
        int w = i % width;
        int idx = n * channels * spatial_size + c * spatial_size + h * width + w;
        float val = input[idx];
        val = hardtanh(val, hardtanh_min, hardtanh_max);
        sdata[tid] += val;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean_val = sdata[0] / spatial_size;
        output[n * channels + c] = tanhf(mean_val);
    }
}

torch::Tensor hardtanh_mean_tanh_cuda(
    torch::Tensor input,
    float hardtanh_min,
    float hardtanh_max
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::empty({batch_size, channels}, input.options());
    
    dim3 blocks(batch_size, channels);
    int threads = min(height * width, 1024);
    int shared_mem = threads * sizeof(float);
    
    hardtanh_mean_tanh_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        hardtanh_min,
        hardtanh_max
    );
    
    return output.unsqueeze(-1).unsqueeze(-1);
}
"""

hardtanh_mean_tanh_cpp_source = """
torch::Tensor hardtanh_mean_tanh_cuda(torch::Tensor input, float hardtanh_min, float hardtanh_max);
"""

hardtanh_mean_tanh = load_inline(
    name="hardtanh_mean_tanh",
    cpp_sources=hardtanh_mean_tanh_cpp_source,
    cuda_sources=hardtanh_mean_tanh_source,
    functions=["hardtanh_mean_tanh_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.hardtanh_mean_tanh = hardtanh_mean_tanh

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.hardtanh_mean_tanh.hardtanh_mean_tanh_cuda(x, self.hardtanh_min, self.hardtanh_max)
        return x