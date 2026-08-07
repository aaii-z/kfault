import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations: MaxPool3d + Softmax + Subtract + Swish + Max
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_post_conv_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ subtract_param, 
    float* __restrict__ output, 
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int pool_kernel_size, int pool_stride, int pool_padding) {
    
    // Dynamic shared memory for reduction
    extern __shared__ float s_data[];
    
    int c = threadIdx.x;
    int idx = blockIdx.x;
    
    // Decode spatial coordinates for the output of MaxPool (which is the final output)
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int n = idx / (W_out * H_out * D_out);
    
    // 1. MaxPool3d
    // Thread c computes max for channel c
    float max_pool_val = -INFINITY;
    
    if (c < C) {
        int d_start = d_out * pool_stride - pool_padding;
        int h_start = h_out * pool_stride - pool_padding;
        int w_start = w_out * pool_stride - pool_padding;
        
        // Iterate over pooling window
        for (int pd = 0; pd < pool_kernel_size; ++pd) {
            for (int ph = 0; ph < pool_kernel_size; ++ph) {
                for (int pw = 0; pw < pool_kernel_size; ++pw) {
                    int d_in = d_start + pd;
                    int h_in = h_start + ph;
                    int w_in = w_start + pw;
                    
                    if (d_in >= 0 && d_in < D_in && 
                        h_in >= 0 && h_in < H_in && 
                        w_in >= 0 && w_in < W_in) {
                        
                        long long linear_idx = (long long)n * C * D_in * H_in * W_in + 
                                               c * D_in * H_in * W_in + 
                                               d_in * H_in * W_in + 
                                               h_in * W_in + 
                                               w_in;
                        float val = input[linear_idx];
                        if (val > max_pool_val) max_pool_val = val;
                    }
                }
            }
        }
        s_data[c] = max_pool_val;
    } else {
        s_data[c] = -INFINITY; // Identity for Max
    }
    __syncthreads();

    // 2. Softmax
    // Reduce Max for numerical stability
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (c < stride) {
            if (s_data[c + stride] > s_data[c]) s_data[c] = s_data[c + stride];
        }
        __syncthreads();
    }
    float max_val = s_data[0];
    __syncthreads();

    float exp_val = 0.0f;
    if (c < C) {
        exp_val = expf(max_pool_val - max_val);
        s_data[c] = exp_val;
    } else {
        s_data[c] = 0.0f; // Identity for Sum
    }
    __syncthreads();

    // Reduce Sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (c < stride) {
            s_data[c] += s_data[c + stride];
        }
        __syncthreads();
    }
    float sum_val = s_data[0];
    
    // 3. Subtract & Swish
    float swish_val = -INFINITY;
    if (c < C) {
        float softmax_val = exp_val / sum_val;
        float sub_val = subtract_param[c];
        float x = softmax_val - sub_val;
        // Swish: sigmoid(x) * x
        swish_val = (1.0f / (1.0f + expf(-x))) * x;
        s_data[c] = swish_val;
    } else {
        s_data[c] = -INFINITY; // Identity for Max
    }
    __syncthreads();

    // 4. Max (final reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (c < stride) {
            if (s_data[c + stride] > s_data[c]) s_data[c] = s_data[c + stride];
        }
        __syncthreads();
    }
    
    if (c == 0) {
        output[idx] = s_data[0];
    }
}

torch::Tensor fused_op_cuda(
    torch::Tensor input, 
    torch::Tensor subtract_param, 
    int pool_kernel_size, int pool_stride, int pool_padding) {
    
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D_in = sizes[2];
    int H_in = sizes[3];
    int W_in = sizes[4];
    
    // Calculate output sizes based on MaxPool3d logic
    int D_out = (D_in + 2 * pool_padding - pool_kernel_size) / pool_stride + 1;
    int H_out = (H_in + 2 * pool_padding - pool_kernel_size) / pool_stride + 1;
    int W_out = (W_in + 2 * pool_padding - pool_kernel_size) / pool_stride + 1;
    
    auto output = torch::zeros({N, D_out, H_out, W_out}, input.options());
    
    int total_pixels = N * D_out * H_out * W_out;
    
    // Block size: next power of 2 >= C
    int block_size = 1;
    while(block_size < C) block_size <<= 1;
    if (block_size > 1024) block_size = 1024; // Cap at max block size
    
    dim3 grid(total_pixels);
    dim3 block(block_size);
    
    size_t shared_mem_size = block_size * sizeof(float);
    
    fused_post_conv_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        subtract_param.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        pool_kernel_size, pool_stride, pool_padding
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor input, torch::Tensor subtract_param, int pool_kernel_size, int pool_stride, int pool_padding);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_op_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        
        self.fused_op = fused_ops.fused_op_cuda

    def forward(self, x):
        x = self.conv_transpose(x)
        # Call custom fused kernel
        x = self.fused_op(x, self.subtract, self.pool_kernel_size, self.pool_stride, self.pool_padding)
        return x