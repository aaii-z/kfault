import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Softmax and Sigmoid
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                       int N, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = N * D * H * W;
    
    if (idx < total_spatial) {
        // Decode linear index to spatial coordinates (n, d, h, w)
        int w = idx % W;
        int temp = idx / W;
        int h = temp % H;
        temp /= H;
        int d = temp % D;
        int n = temp / D;
        
        // Calculate strides for NCDHW layout
        long long stride_n = (long long)C * D * H * W;
        long long stride_c = (long long)D * H * W;
        long long stride_d = (long long)H * W;
        long long stride_h = (long long)W;
        long long stride_w = 1;
        
        // Base offset for the start of the channel vector at (n, d, h, w)
        long long base_offset = n * stride_n + d * stride_d + h * stride_h + w * stride_w;
        
        // 1. Find Max for numerical stability
        float max_val = -INFINITY;
        for (int c = 0; c < C; ++c) {
            float val = input[base_offset + c * stride_c];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        // 2. Compute Sum of exp(x - max)
        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = input[base_offset + c * stride_c];
            sum_exp += expf(val - max_val);
        }
        
        // 3. Compute Softmax and Sigmoid, write output
        for (int c = 0; c < C; ++c) {
            long long offset = base_offset + c * stride_c;
            float val = input[offset];
            float softmax_val = expf(val - max_val) / sum_exp;
            float sigmoid_val = 1.0f / (1.0f + expf(-softmax_val));
            output[offset] = sigmoid_val;
        }
    }
}

torch::Tensor softmax_sigmoid_cuda(torch::Tensor x) {
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    
    auto output = torch::empty_like(x);
    
    int total_spatial = N * D * H * W;
    const int block_size = 256;
    const int num_blocks = (total_spatial + block_size - 1) / block_size;
    
    softmax_sigmoid_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), N, C, D, H, W
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor softmax_sigmoid_cuda(torch::Tensor x);"

# Compile the inline CUDA code
softmax_sigmoid_op = load_inline(
    name="softmax_sigmoid_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["softmax_sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax_sigmoid = softmax_sigmoid_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.softmax_sigmoid.softmax_sigmoid_cuda(x)
        return x