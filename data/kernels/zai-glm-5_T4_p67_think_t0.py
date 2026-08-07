import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GELU and Global Average Pooling
gelu_gap_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_gap_kernel(const float* __restrict__ input, float* __restrict__ output, int B, int C, int H, int W) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    int spatial_size = H * W;
    float sum = 0.0f;
    
    // Calculate base pointer for this (b, c)
    // Input shape: (B, C, H, W)
    // Offset: (b * C + c) * H * W
    const float* input_ptr = input + (static_cast<long long>(b) * C + c) * spatial_size;
    
    // Vectorized loading for coalesced memory access
    // We process 4 floats at a time
    int spatial_size_4 = spatial_size / 4;
    const float4* input_4 = reinterpret_cast<const float4*>(input_ptr);
    
    for (int i = tid; i < spatial_size_4; i += block_size) {
        float4 val4 = input_4[i];
        // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        // 1/sqrt(2) approx 0.7071067811865475
        sum += val4.x * 0.5f * (1.0f + erff(val4.x * 0.7071067811865475f));
        sum += val4.y * 0.5f * (1.0f + erff(val4.y * 0.7071067811865475f));
        sum += val4.z * 0.5f * (1.0f + erff(val4.z * 0.7071067811865475f));
        sum += val4.w * 0.5f * (1.0f + erff(val4.w * 0.7071067811865475f));
    }
    
    // Handle remaining elements if spatial_size is not divisible by 4
    int remaining_start = spatial_size_4 * 4;
    for (int i = remaining_start + tid; i < spatial_size; i += block_size) {
        float val = input_ptr[i];
        sum += val * 0.5f * (1.0f + erff(val * 0.7071067811865475f));
    }
    
    // Block reduction using shared memory
    __shared__ float shared_sum[256]; // Assuming block size 256
    shared_sum[tid] = sum;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[b * C + c] = shared_sum[0] / spatial_size;
    }
}

torch::Tensor gelu_gap_cuda(torch::Tensor input) {
    // Input shape: (B, C, H, W)
    // Output shape: (B, C)
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    auto output = torch::zeros({B, C}, input.options());
    
    const int block_size = 256;
    dim3 grid(B, C);
    
    gelu_gap_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        B, C, H, W
    );
    
    return output;
}
"""

gelu_gap_cpp_source = (
    "torch::Tensor gelu_gap_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code
gelu_gap_op = load_inline(
    name="gelu_gap",
    cpp_sources=gelu_gap_cpp_source,
    cuda_sources=gelu_gap_source,
    functions=["gelu_gap_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.gelu_gap = gelu_gap_op

    def forward(self, x):
        x = self.conv(x)
        # Ensure contiguous memory for the custom kernel
        x = self.gelu_gap.gelu_gap_cuda(x.contiguous())
        return x