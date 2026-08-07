import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused softmax, bias addition, scaling, and sigmoid
# This kernel is optimized to avoid the timeout by using a standard parallel reduction approach
# and efficient memory access patterns.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// Kernel for Softmax + Bias + Scale + Sigmoid
// Optimized for channel-first layout (N, C, H, W)
// Uses a parallel reduction approach within each block.
__global__ void fused_ops_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out, int N, int C, int H, int W, float scaling_factor) {
    // Each block processes one (n, h, w) spatial location across all channels
    int n = blockIdx.x / (H * W);
    int hw_rem = blockIdx.x % (H * W);
    int h = hw_rem / W;
    int w = hw_rem % W;

    // Base index for the channel dimension at this spatial location
    int base_idx = n * C * H * W + h * W + w;
    
    // Stride for accessing channels
    int stride = H * W;

    // Shared memory for reduction
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;

    // 1. Find Max
    // Each thread finds max in its assigned portion of the channel dimension
    float max_val = -INFINITY;
    for (int c = tid; c < C; c += blockDim.x) {
        float val = x[base_idx + c * stride];
        if (val > max_val) {
            max_val = val;
        }
    }
    shared_max[tid] = max_val;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }
    max_val = shared_max[0];
    __syncthreads();

    // 2. Calculate Sum of Exponentials
    float sum_val = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        float val = x[base_idx + c * stride];
        sum_val += expf(val - max_val);
    }
    shared_sum[tid] = sum_val;
    __syncthreads();

    // Parallel reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    sum_val = shared_sum[0];
    __syncthreads();

    // 3. Normalize, Add Bias, Scale, Sigmoid
    // Softmax(x) = exp(x - max) / sum
    // Result = sigmoid(softmax * scaling_factor + bias)
    for (int c = tid; c < C; c += blockDim.x) {
        int idx = base_idx + c * stride;
        float val = x[idx];
        float softmax_val = expf(val - max_val) / sum_val;
        float bias_val = bias[c]; // bias shape is (C, 1, 1)
        float res = softmax_val * scaling_factor + bias_val;
        out[idx] = 1.0f / (1.0f + expf(-res));
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias, float scaling_factor) {
    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    
    auto out = torch::empty_like(x);
    
    // Grid: one block per spatial location
    int num_blocks = N * H * W;
    int threads_per_block = 256; // Assuming C <= 1024, 256 is a good balance
    
    fused_ops_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        scaling_factor
    );
    
    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor bias, float scaling_factor);"
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.scaling_factor)
        return x