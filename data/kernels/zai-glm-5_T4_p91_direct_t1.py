import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused softmax, bias addition, scaling, and sigmoid
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel for Softmax + Bias + Scale + Sigmoid
__global__ void fused_ops_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out, int N, int C, int H, int W, float scaling_factor) {
    // Each block processes one (n, h, w) spatial location across all channels
    // Block size is assumed to be 256 threads
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
    int lane_id = tid & 31;

    // 1. Find Max
    float max_val = -INFINITY;
    for (int c = tid; c < C; c += blockDim.x) {
        float val = x[base_idx + c * stride];
        if (val > max_val) max_val = val;
    }
    
    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    
    // Store warp results in shared memory
    if (lane_id == 0) shared_max[tid / 32] = max_val;
    __syncthreads();

    // Final reduction by first warp
    if (tid < 8) {
        max_val = shared_max[tid];
        for (int offset = 4; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (tid == 0) shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];

    // 2. Compute Exp and Sum
    float sum_val = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        float val = expf(x[base_idx + c * stride] - max_val);
        // We can write back the exp result to 'out' temporarily or keep in register if memory allows
        // Writing to out saves re-computation
        out[base_idx + c * stride] = val;
        sum_val += val;
    }

    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
    }

    if (lane_id == 0) shared_sum[tid / 32] = sum_val;
    __syncthreads();

    if (tid < 8) {
        sum_val = shared_sum[tid];
        for (int offset = 4; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
        if (tid == 0) shared_sum[0] = sum_val;
    }
    __syncthreads();
    sum_val = shared_sum[0];

    // 3. Normalize, Add Bias, Scale, Sigmoid
    // Softmax(x) = exp(x - max) / sum
    // Result = sigmoid(softmax * scaling_factor + bias)
    for (int c = tid; c < C; c += blockDim.x) {
        int idx = base_idx + c * stride;
        float softmax_val = out[idx] / sum_val;
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