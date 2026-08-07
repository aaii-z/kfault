import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused softmax, bias addition, scaling, and sigmoid
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for Softmax + Bias + Scale + Sigmoid
// Optimized for channel counts around 128. 
// Uses shared memory for reduction. 
// Each block processes one (n, h, w) spatial location across all channels.
__global__ void fused_ops_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out, int N, int C, int H, int W, float scaling_factor) {
    int n = blockIdx.x / (H * W);
    int hw_rem = blockIdx.x % (H * W);
    int h = hw_rem / W;
    int w = hw_rem % W;

    int base_idx = n * C * H * W + h * W + w;
    int stride = H * W;

    // Shared memory for reduction
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;

    // 1. Find Max
    float max_val = -INFINITY;
    // Loop over channels with stride blockDim.x
    for (int c = tid; c < C; c += blockDim.x) {
        float val = x[base_idx + c * stride];
        if (val > max_val) max_val = val;
    }
    
    // Parallel reduction in shared memory
    shared_max[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    max_val = shared_max[0];
    __syncthreads();

    // 2. Compute Exp and Sum
    float sum_val = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        float val = expf(x[base_idx + c * stride] - max_val);
        // Store exp result temporarily in out to save registers/recompute
        out[base_idx + c * stride] = val;
        sum_val += val;
    }

    shared_sum[tid] = sum_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    sum_val = shared_sum[0];
    __syncthreads();

    // 3. Normalize, Add Bias, Scale, Sigmoid
    float inv_sum = 1.0f / sum_val;
    
    for (int c = tid; c < C; c += blockDim.x) {
        int idx = base_idx + c * stride;
        float softmax_val = out[idx] * inv_sum;
        float bias_val = bias[c]; // bias shape is (C, 1, 1)
        float val = softmax_val * scaling_factor + bias_val;
        out[idx] = 1.0f / (1.0f + expf(-val));
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
    int threads_per_block = 256; 
    
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