import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused maxpool and clamp
maxpool_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void maxpool_clamp_kernel(const float* input, float* output, 
                                     int batch_size, int channels, int height, int width,
                                     int pool_size, float clamp_min, float clamp_max) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h_out = blockIdx.z * blockDim.z + threadIdx.z;
    int w_out = threadIdx.x;

    int h_stride = height / pool_size;
    int w_stride = width / pool_size;
    
    if (h_out >= h_stride || w_out >= w_stride) return;

    float max_val = -FLT_MAX;
    for (int dh = 0; dh < pool_size; ++dh) {
        for (int dw = 0; dw < pool_size; ++dw) {
            int h_in = h_out * pool_size + dh;
            int w_in = w_out * pool_size + dw;
            int idx = n * channels * height * width + c * height * width + h_in * width + w_in;
            max_val = fmaxf(max_val, input[idx]);
        }
    }
    max_val = fmaxf(clamp_min, fminf(max_val, clamp_max));
    
    int out_idx = n * channels * h_stride * w_stride + c * h_stride * w_stride + h_out * w_stride + w_out;
    output[out_idx] = max_val;
}

torch::Tensor maxpool_clamp_cuda(torch::Tensor input, int pool_size, float clamp_min, float clamp_max) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int h_stride = height / pool_size;
    int w_stride = width / pool_size;
    auto output = torch::empty({batch_size, channels, h_stride, w_stride}, input.options());
    
    dim3 blocks(batch_size, channels, h_stride);
    dim3 threads(w_stride, 1, 1);
    
    maxpool_clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, height, width,
        pool_size, clamp_min, clamp_max
    );
    
    return output;
}
"""

maxpool_clamp_cpp_source = "torch::Tensor maxpool_clamp_cuda(torch::Tensor input, int pool_size, float clamp_min, float clamp_max);"

# Compile the inline CUDA code
maxpool_clamp = load_inline(
    name="maxpool_clamp",
    cpp_sources=maxpool_clamp_cpp_source,
    cuda_sources=maxpool_clamp_source,
    functions=["maxpool_clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Fuse scale into group norm parameters
        scale = torch.ones(scale_shape)
        self.group_norm.weight = nn.Parameter(self.group_norm.weight * scale.view(-1))
        self.group_norm.bias = nn.Parameter(self.group_norm.bias * scale.view(-1))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = maxpool_clamp.maxpool_clamp_cuda(x, self.maxpool_kernel_size, self.clamp_min, self.clamp_max)
        return x