import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused subtract + HardSwish
fused_sub_hardswish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_sub_hardswish_kernel(
    const float* input, float* output, float subtract_value, int num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx] - subtract_value;
        output[idx] = val * fminf(fmaxf(0.0f, val + 3.0f), 6.0f) / 6.0f;
    }
}

torch::Tensor fused_sub_hardswish_cuda(torch::Tensor input, float subtract_value) {
    CHECK_INPUT(input);
    
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    const int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    
    fused_sub_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), subtract_value, num_elements);
    
    return output;
}
"""

fused_sub_hardswish_cpp = "torch::Tensor fused_sub_hardswish_cuda(torch::Tensor, float);"

# Custom CUDA kernel for fused MaxPool + Mish
fused_pool_mish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ float mish(float x) {
    return x * tanh(log(1.0f + exp(x)));
}

__global__ void fused_pool_mish_kernel(
    const float* input, float* output, 
    int batch_size, int channels, int height, int width,
    int pool_size, int output_h, int output_w) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_h * output_w;
    if (index >= total_elements) return;
    
    int n = index / (channels * output_h * output_w);
    int c = (index % (channels * output_h * output_w)) / (output_h * output_w);
    int h = (index % (output_h * output_w)) / output_w;
    int w = index % output_w;
    
    int h_start = h * pool_size;
    int w_start = w * pool_size;
    int h_end = min(h_start + pool_size, height);
    int w_end = min(w_start + pool_size, width);
    
    float max_val = -FLT_MAX;
    for (int ih = h_start; ih < h_end; ++ih) {
        for (int iw = w_start; iw < w_end; ++iw) {
            int input_idx = ((n * channels + c) * height + ih) * width + iw;
            max_val = fmaxf(max_val, input[input_idx]);
        }
    }
    
    output[index] = mish(max_val);
}

torch::Tensor fused_pool_mish_cuda(torch::Tensor input, int pool_size) {
    CHECK_INPUT(input);
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int output_h = (height - pool_size) / pool_size + 1;
    int output_w = (width - pool_size) / pool_size + 1;
    
    auto output = torch::empty({batch_size, channels, output_h, output_w}, input.options());
    
    int total_elements = batch_size * channels * output_h * output_w;
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_pool_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, height, width,
        pool_size, output_h, output_w);
    
    return output;
}
"""

fused_pool_mish_cpp = "torch::Tensor fused_pool_mish_cuda(torch::Tensor, int);"

# Load custom CUDA extensions
fused_sub_hardswish = load_inline(
    name='fused_sub_hardswish',
    cpp_sources=fused_sub_hardswish_cpp,
    cuda_sources=fused_sub_hardswish_source,
    functions=['fused_sub_hardswish_cuda'],
    verbose=True
)

fused_pool_mish = load_inline(
    name='fused_pool_mish',
    cpp_sources=fused_pool_mish_cpp,
    cuda_sources=fused_pool_mish_source,
    functions=['fused_pool_mish_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = fused_sub_hardswish.fused_sub_hardswish_cuda(x, self.subtract_value)
        x = fused_pool_mish.fused_pool_mish_cuda(x, self.pool_kernel_size)
        return x