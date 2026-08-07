import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: Scale, MaxPool3d, GlobalAvgPool3d, Clamp
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <float.h>

// Kernel to fuse: Scale -> MaxPool3d -> GlobalAvgPool3d -> Clamp
// Input: (N, C, D_in, H_in, W_in)
// Output: (N, C, 1, 1, 1)
__global__ void fused_ops_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int N, int C, 
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int pool_size, // Assuming stride = pool_size
    float scale, float clamp_min, float clamp_max) {
    
    int nc = blockIdx.x;
    if (nc >= N * C) return;

    int n = nc / C;
    int c = nc % C;
    
    const float* in_ptr = input + n * C * D_in * H_in * W_in + c * D_in * H_in * W_in;
    
    int total_spatial = D_out * H_out * W_out;
    
    // Shared memory for reduction
    extern __shared__ float s_sum[];
    
    float local_sum = 0.0f;
    
    // Grid-stride loop to cover all spatial output positions of MaxPool
    for (int idx = threadIdx.x; idx < total_spatial; idx += blockDim.x) {
        // Calculate output coordinates (d_out, h_out, w_out)
        int d_out = idx / (H_out * W_out);
        int rem = idx % (H_out * W_out);
        int h_out = rem / W_out;
        int w_out = rem % W_out;
        
        // Calculate input window start
        int d_start = d_out * pool_size;
        int h_start = h_out * pool_size;
        int w_start = w_out * pool_size;
        
        // Max Pooling
        float max_val = -FLT_MAX;
        
        // Loops for pooling window
        for (int pd = 0; pd < pool_size; ++pd) {
            for (int ph = 0; ph < pool_size; ++ph) {
                for (int pw = 0; pw < pool_size; ++pw) {
                    int d_in = d_start + pd;
                    int h_in = h_start + ph;
                    int w_in = w_start + pw;
                    
                    int in_idx = d_in * H_in * W_in + h_in * W_in + w_in;
                    float val = in_ptr[in_idx] * scale;
                    if (val > max_val) max_val = val;
                }
            }
        }
        local_sum += max_val;
    }
    
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float avg = s_sum[0] / total_spatial;
        // Clamp
        if (avg < clamp_min) avg = clamp_min;
        if (avg > clamp_max) avg = clamp_max;
        output[nc] = avg;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input, 
    int64_t pool_size, 
    double scale, 
    double clamp_min, 
    double clamp_max) {
    
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    // Calculate output dimensions of MaxPool
    // Assuming stride = pool_size, padding = 0
    int D_out = (D_in - pool_size) / pool_size + 1;
    int H_out = (H_in - pool_size) / pool_size + 1;
    int W_out = (W_in - pool_size) / pool_size + 1;
    
    auto output = torch::zeros({N, C, 1, 1, 1}, input.options());
    
    int threads = 256;
    int blocks = N * C;
    int shared_mem = threads * sizeof(float);
    
    fused_ops_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        (int)pool_size,
        (float)scale, (float)clamp_min, (float)clamp_max
    );
    
    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor input, int64_t pool_size, double scale, double clamp_min, double clamp_max);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed 3D convolution, followed by a fused
    CUDA kernel performing scalar multiplication, max pooling, global average pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused operation: Scale -> MaxPool -> GlobalAvgPool -> Clamp
        x = self.fused_ops.fused_ops_cuda(x, self.maxpool_kernel_size, self.scale, self.clamp_min, self.clamp_max)
        return x