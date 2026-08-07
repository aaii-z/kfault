import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused MaxPool3d + LogSumExp + ReLU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride_n, const int stride_c, const int stride_d, const int stride_h, const int stride_w) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D_out * H_out * W_out;
    
    if (index >= total_elements) return;

    // Calculate output coordinates (n, d, h, w)
    int w_out = index % W_out;
    int h_out = (index / W_out) % H_out;
    int d_out = (index / (W_out * H_out)) % D_out;
    int n = index / (W_out * H_out * D_out);

    // Calculate input top-left corner for the pooling window
    int d_in = d_out * 2;
    int h_in = h_out * 2;
    int w_in = w_out * 2;

    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // Iterate over channels
    for (int c = 0; c < C; ++c) {
        const float* ptr_base = input + n * stride_n + c * stride_c + d_in * stride_d + h_in * stride_h + w_in * stride_w;
        
        float pool_max = -INFINITY;
        
        // Unrolled 2x2x2 MaxPool
        // We use fmaxf for performance
        float v;
        
        // z=0
        v = ptr_base[0]; pool_max = fmaxf(pool_max, v);
        v = ptr_base[stride_w]; pool_max = fmaxf(pool_max, v);
        v = ptr_base[stride_h]; pool_max = fmaxf(pool_max, v);
        v = ptr_base[stride_h + stride_w]; pool_max = fmaxf(pool_max, v);
        
        // z=1
        v = ptr_base[stride_d]; pool_max = fmaxf(pool_max, v);
        v = ptr_base[stride_d + stride_w]; pool_max = fmaxf(pool_max, v);
        v = ptr_base[stride_d + stride_h]; pool_max = fmaxf(pool_max, v);
        v = ptr_base[stride_d + stride_h + stride_w]; pool_max = fmaxf(pool_max, v);
        
        // Online LogSumExp algorithm
        float new_max = fmaxf(max_val, pool_max);
        
        // Handle the case where new_max is -infinity (all values are -inf)
        if (new_max > -INFINITY) {
            // If max_val was -inf, scale is 0 (exp(-inf)), sum_exp remains 0
            // Otherwise scale is exp(max_val - new_max)
            float scale = (max_val > -INFINITY) ? expf(max_val - new_max) : 0.0f;
            sum_exp = sum_exp * scale + expf(pool_max - new_max);
            max_val = new_max;
        }
    }
    
    float result = logf(sum_exp) + max_val;
    
    // ReLU
    if (result < 0.0f) result = 0.0f;
    
    output[index] = result;
}

torch::Tensor fused_maxpool_lse_relu_cuda(torch::Tensor input) {
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D_in = sizes[2];
    int H_in = sizes[3];
    int W_in = sizes[4];
    
    // MaxPool3d with kernel_size=2, stride=2
    int D_out = D_in / 2;
    int H_out = H_in / 2;
    int W_out = W_in / 2;
    
    auto output = torch::zeros({N, 1, D_out, H_out, W_out}, input.options());
    
    int total_elements = N * D_out * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    auto strides = input.strides();
    
    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out,
        strides[0], strides[1], strides[2], strides[3], strides[4]
    );
    
    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_maxpool_lse_relu_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_maxpool_lse_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_maxpool_lse_relu_cuda(x)
        return x