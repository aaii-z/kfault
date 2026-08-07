import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused subtract + HardSwish
fused_sub_hardswish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_sub_hardswish_kernel(
    const float* input, float* output, float subtract_value, int num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx] - subtract_value;
        // HardSwish: x * min(max(0, x+3), 6) / 6
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

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ float mish(float x) {
    return x * tanh(log(1.0f + exp(x)));
}

__global__ void fused_pool_mish_kernel(
    const float* input, float* output, 
    int batch_size, int channels, int height, int width,
    int pool_size, int pool_stride) {
    
    int output_h = (height - pool_size) / pool_stride + 1;
    int output_w = (width - pool_size) / pool_stride + 1;
    
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = threadIdx.x;
    int w = threadIdx.y;
    
    if (h >= output_h || w >= output_w) return;
    
    float max_val = -FLT_MAX;
    for (int ph = 0; ph < pool_size; ++ph) {
        for (int pw = 0; pw < pool_size; ++pw) {
            int ih = h * pool_stride + ph;
            int iw = w * pool_stride + pw;
            if (ih < height && iw < width) {
                int idx = ((n * channels + c) * height + ih) * width + iw;
                max_val = fmaxf(max_val, input[idx]);
            }
        }
    }
    
    int output_idx = ((n * channels + c) * output_h + h) * output_w + w;
    output[output_idx] = mish(max_val);
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
    
    dim3 blocks(batch_size, channels);
    dim3 threads(output_h, output_w);
    
    fused_pool_mish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, height, width,
        pool_size, pool_size);
    
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