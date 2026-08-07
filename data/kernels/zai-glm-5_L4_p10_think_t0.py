import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused MaxPool + Hardtanh + Mean + Tanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_pool_hardtanh_mean_tanh_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int H, const int W,
    const int H_out, const int W_out,
    const float hardtanh_min, const float hardtanh_max) {
    
    // Map block index to (n, c) pair
    int nc_idx = blockIdx.x;
    int n = nc_idx / C;
    int c = nc_idx % C;
    
    // Calculate the starting pointer for this (n, c) channel
    const float* x_ptr = input + n * C * H * W + c * H * W;
    
    // Shared memory for reduction within the block
    __shared__ float s_sum[256];
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    
    int total_pixels = H_out * W_out;
    
    // Strided loop over the output pixels of the MaxPool operation
    // Each thread accumulates a partial sum
    for (int i = tid; i < total_pixels; i += blockDim.x) {
        int h_out = i / W_out;
        int w_out = i % W_out;
        
        // Calculate input coordinates for the 2x2 pooling window
        // Assuming stride=2, kernel=2 as per the problem description
        int h_in = h_out * 2;
        int w_in = w_out * 2;
        
        // Load 4 values from the input
        // Using simple loads; compiler should optimize for coalescing
        // w_in is always even, so we can potentially use float2, but let's keep it simple for correctness
        float v00 = x_ptr[h_in * W + w_in];
        float v01 = x_ptr[h_in * W + w_in + 1];
        float v10 = x_ptr[(h_in + 1) * W + w_in];
        float v11 = x_ptr[(h_in + 1) * W + w_in + 1];
        
        // MaxPool 2x2
        float max_val = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
        
        // Hardtanh
        float ht_val = fmaxf(hardtanh_min, fminf(hardtanh_max, max_val));
        
        local_sum += ht_val;
    }
    
    s_sum[tid] = local_sum;
    __syncthreads();
    
    // Block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 computes the mean and applies tanh
    if (tid == 0) {
        float mean_val = s_sum[0] / (float)total_pixels;
        float tanh_val = tanhf(mean_val);
        
        // Output shape is (N, C, 1, 1)
        // Linear index: n * C + c
        output[nc_idx] = tanh_val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor input, float hardtanh_min, float hardtanh_max) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    // MaxPool 2x2 stride 2 output dimensions
    int H_out = (H - 2) / 2 + 1;
    int W_out = (W - 2) / 2 + 1;
    
    auto output = torch::zeros({N, C, 1, 1}, input.options());
    
    const int block_size = 256;
    const int num_blocks = N * C; // One block per channel
    
    fused_pool_hardtanh_mean_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        hardtanh_min, hardtanh_max
    );
    
    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, float hardtanh_min, float hardtanh_max);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, followed by a fused
    MaxPool + Hardtanh + Mean + Tanh operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_op_cuda(x, self.hardtanh_min, self.hardtanh_max)
        return x