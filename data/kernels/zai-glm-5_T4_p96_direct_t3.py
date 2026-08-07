import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Kernel for fused scale, maxpool, global avg pool, and clamp
// Optimized for non-overlapping pooling (stride == kernel_size)
__global__ void fused_ops_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    float scale,
    int N, int C, int iD, int iH, int iW,
    int pool_kD, int pool_kH, int pool_kW
) {
    // Each block handles one (N, C) channel
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc % C;

    // Calculate dimensions after maxpool
    // Assuming stride = kernel_size and padding = 0
    int mpD = iD / pool_kD;
    int mpH = iH / pool_kH;
    int mpW = iW / pool_kW;
    int mp_size = mpD * mpH * mpW;

    // Use shared memory to store maxpool results for this channel
    extern __shared__ float shared_mp[];
    
    // Initialize shared memory for maxpool
    for (int i = threadIdx.x; i < mp_size; i += blockDim.x) {
        shared_mp[i] = -FLT_MAX;
    }
    __syncthreads();

    // Pointers
    const float* __restrict__ in_ptr = input + n * (C * iD * iH * iW) + c * (iD * iH * iW);

    // Grid-stride loop over input for MaxPool
    int total_input_pixels = iD * iH * iW;

    for (int idx = threadIdx.x; idx < total_input_pixels; idx += blockDim.x) {
        int d = idx / (iH * iW);
        int rem = idx % (iH * iW);
        int h = rem / iW;
        int w = rem % iW;

        // Apply scale
        float val = in_ptr[idx] * scale;

        // Calculate output position in maxpool (non-overlapping)
        int od = d / pool_kD;
        int oh = h / pool_kH;
        int ow = w / pool_kW;

        int out_idx = od * mpH * mpW + oh * mpW + ow;

        // Atomic max
        // Using integer comparison for float atomicMax
        int* shared_mp_int = (int*)shared_mp;
        int val_int = __float_as_int(val);
        atomicMax(&shared_mp_int[out_idx], val_int);
    }
    __syncthreads();

    // Global Average Pooling on shared_mp
    // Sum reduction in shared memory
    float sum = 0.0f;
    for (int i = threadIdx.x; i < mp_size; i += blockDim.x) {
        sum += shared_mp[i];
    }

    // Block reduce sum
    // Using a fixed size buffer for reduction
    __shared__ float block_sum[256]; 
    block_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float avg = block_sum[0] / (float)mp_size;
        // Clamp
        if (avg < 0.0f) avg = 0.0f;
        if (avg > 1.0f) avg = 1.0f;
        output[nc] = avg;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, 
    float scale,
    int pool_kD, int pool_kH, int pool_kW
) {
    int N = input.size(0);
    int C = input.size(1);
    int iD = input.size(2);
    int iH = input.size(3);
    int iW = input.size(4);

    auto output = torch::zeros({N, C}, input.options());

    int blocks = N * C;
    int threads = 256; // Good balance
    
    // Calculate shared memory size
    int mpD = iD / pool_kD;
    int mpH = iH / pool_kH;
    int mpW = iW / pool_kW;
    int shared_mem_size = mpD * mpH * mpW * sizeof(float);

    fused_ops_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale,
        N, C, iD, iH, iW,
        pool_kD, pool_kH, pool_kW
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, float scale, "
    "int pool_kD, int pool_kH, int pool_kW);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused scale, maxpool, global avg pool, clamp
        # Assuming MaxPool3d parameters: kernel_size=maxpool_kernel_size, stride=kernel_size, padding=0 (default)
        k = self.maxpool_kernel_size
        x = self.fused_ops.fused_ops_cuda(x, self.scale, k, k, k)
        return x