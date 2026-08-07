import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused AvgPool + Sigmoid + Sum
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_pool_sigmoid_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pool_size) {

    int n = blockIdx.x;
    int c = blockIdx.y;
    
    const float* in_ptr = input + (n * C + c) * H_in * W_in;
    
    __shared__ float s_data[256];
    
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    
    int total_elements = H_out * W_out;
    int pool_area = pool_size * pool_size;
    
    for (int i = tid; i < total_elements; i += blockDim.x) {
        int h_out = i / W_out;
        int w_out = i % W_out;
        
        int h_in_start = h_out * pool_size;
        int w_in_start = w_out * pool_size;
        
        float pool_val = 0.0f;
        
        // Compute average pool
        for (int kh = 0; kh < pool_size; ++kh) {
            int h_in = h_in_start + kh;
            for (int kw = 0; kw < pool_size; ++kw) {
                int w_in = w_in_start + kw;
                pool_val += in_ptr[h_in * W_in + w_in];
            }
        }
        
        pool_val /= pool_area;
        
        // Sigmoid
        float sig_val = 1.0f / (1.0f + expf(-pool_val));
        
        local_sum += sig_val;
    }
    
    s_data[tid] = local_sum;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&output[n], s_data[0]);
    }
}

torch::Tensor fused_pool_sigmoid_sum_cuda(torch::Tensor input, int pool_size) {
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    // Calculate output dimensions
    int H_out = (H_in - pool_size) / pool_size + 1;
    int W_out = (W_in - pool_size) / pool_size + 1;
    
    auto output = torch::zeros({N}, input.options());
    
    const int block_size = 256;
    dim3 grid(N, C);
    dim3 block(block_size);
    
    fused_pool_sigmoid_sum_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out, pool_size
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor fused_pool_sigmoid_sum_cuda(torch::Tensor input, int pool_size);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_pool_sigmoid_sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_pool_sigmoid_sum_cuda(x, self.pool_kernel_size)
        return x