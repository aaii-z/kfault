import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_hardtanh_mean_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

__global__ void fused_hardtanh_mean_tanh_kernel(
    const float* x,
    float* out,
    int B,
    int C,
    int H,
    int W,
    float hardtanh_min,
    float hardtanh_max) {
    
    int c = blockIdx.x;
    int b = blockIdx.y;
    if (b >= B || c >= C) return;

    int total_elements = H * W;
    float sum = 0.0f;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int i = tid; i < total_elements; i += num_threads) {
        int h = i / W;
        int w = i % W;
        int idx = ((b * C + c) * H + h) * W + w;
        float val = x[idx];
        
        float clamped = val;
        if (clamped > hardtanh_max) clamped = hardtanh_max;
        else if (clamped < hardtanh_min) clamped = hardtanh_min;
        
        sum += clamped;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = shared_sum[0];
        float mean = total_sum / total_elements;
        float result = tanhf(mean);
        out[b * C + c] = result;
    }
}

torch::Tensor fused_hardtanh_mean_tanh_cuda(torch::Tensor x, float hardtanh_min, float hardtanh_max) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be FP32");
    
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    auto output = torch::empty({B, C, 1, 1}, x.options());
    
    dim3 grid(C, B);
    int block_size = 256;
    fused_hardtanh_mean_tanh_kernel<<<grid, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        hardtanh_min, hardtanh_max
    );
    
    return output;
}
"""

fused_hardtanh_mean_tanh_cpp = "torch::Tensor fused_hardtanh_mean_tanh_cuda(torch::Tensor x, float hardtanh_min, float hardtanh_max);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_hardtanh_mean_tanh_cpp,
    cuda_sources=fused_hardtanh_mean_tanh_source,
    functions=["fused_hardtanh_mean_tanh_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = x.contiguous()
        x = self.fused_op.fused_hardtanh_mean_tanh_cuda(x, self.hardtanh_min, self.hardtanh_max)
        return x